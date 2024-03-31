import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset

from model import build_transformer
from dataset_utils import generate_dataset
from dataset import SequencesDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weight_file_path, get_results_path

from torch.utils.tensorboard import SummaryWriter

import warnings
from pathlib import Path
from tqdm import tqdm
import os
import json
import numpy as np

def inference_decoding(model, source, source_mask, seq_len, device):

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)  #(Batch, src_len, d_model)
    # Initialize the decoder input
    decoder_input = torch.zeros(encoder_output.shape[0], 1, 10).type_as(source).to(device)
    while True: # Autoregressive generation
        if decoder_input.size(1) == seq_len:
            break
        
        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).bool().to(device) 

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next element in the sequence
        next_element = model.project(out[:, -1]) # è già sul device? in caso rimuovere il .to(device) che c'è sotto
        decoder_input = torch.cat([decoder_input, next_element.unsqueeze(1).to(device)], dim = 1) # da capire la dimensione di next_element
    
    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, seq_len, device, print_msg, global_step, writer, epoch, results):
    
    model.eval()

    losses = []

    try:
        with os.popen("stty size", "r") as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80
    # i = 1
    batch_iterator = tqdm(validation_ds, desc=f"Processing Validation for Epoch {epoch:02d}")
    with torch.no_grad():
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = None # default for bidirectional encoding            
            model_out = inference_decoding(model, encoder_input, encoder_mask, seq_len, device)

            tgt_seqs = batch["label"]
            pred_seqs = model_out.detach().cpu()

            # Evaluate the model predictions
            mse_loss = nn.MSELoss()
            mse = mse_loss(torch.tensor(tgt_seqs), torch.tensor(pred_seqs))
            batch_iterator.set_postfix({"val loss": f"{mse.item():6.3f}"})
            # print(f"batch {i}: {mse}")
            # i+=1
            losses.append(mse)

        val_loss = sum(losses)/len(losses)
        print(f"val loss: {val_loss.item():6.3f}")
                
        # store the mean loss into results
        results["val_loss"].append(val_loss.item())

        writer.add_scalar(f"{print_msg} MSE", val_loss, global_step)
        writer.flush()

    return results


def get_ds(config):
    print("Checking for existing dataset...")
    dataset_path = f"datasets/{config['security']}_{config['start_year']}-{config['end_year']}_{config['src_len']}-{config['tgt_len']}.npy"
    if os.path.exists(dataset_path):
        print("Found existing dataset at ", dataset_path, ". Loading...")
        with open(dataset_path, "rb") as f:
            X_raw = np.load(f)
            y_raw = np.load(f)
        print("Dataset loaded")
    else:
        print("Dataset not found, generating...")
        X_raw, y_raw = generate_dataset(config["security"], config["provider"], config["src_len"], config["tgt_len"], config["start_year"], config["end_year"]) # generate raw dataset
        with open(dataset_path, "wb") as f:
            np.save(f, X_raw)
            np.save(f, y_raw)
        print("Generated and saved dataset at ", dataset_path)
            
    full_ds = SequencesDataset(X_raw, y_raw, config["src_len"], config["tgt_len"]) # generate dataset object

    # 80% training, 10% validation, 10% test
    train_ds_size = int(0.8 * len(full_ds))
    test_ds_size = int(0.1 * len(full_ds))
    val_ds_size = len(full_ds) - train_ds_size - test_ds_size 

    train_ds, test_ds, val_ds = random_split(full_ds, [train_ds_size, test_ds_size, val_ds_size], generator=torch.Generator().manual_seed(42))

    if config["max_train_size"]:
        train_ds = Subset(train_ds, range(config["max_train_size"]))
    if config["max_val_size"]:
        val_ds = Subset(val_ds, range(config["max_val_size"]))

    train_dataloader = DataLoader(train_ds, batch_size=config["train_batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size = config["val_batch_size"], shuffle = True)
    test_dataloader = DataLoader(test_ds, batch_size = 1, shuffle=True) # batch = 1 è strano in inferenza

    return train_dataloader, test_dataloader, val_dataloader

def get_model(config):
    model = build_transformer(config["src_len"], config["tgt_len"], config["input_features"], config["y_size"], config["d_model"], config["N"], config["h"], config["dropout"], config["d_ff"])
    return model

def train_model(config):

    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, test_dataloader, val_dataloader = get_ds(config)
    model = get_model(config).to(device)

    # Tensorboard
    writer = SummaryWriter(f"{config['security']}_{config['model_folder']}/{config['experiment_name']}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"], eps = 1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config["preload"]
    model_filename = latest_weight_file_path(config) if preload == "latest" else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
        print("Loading results.json file")
        with open(get_results_path(config), "r") as f:
            results = json.load(f)
    else:
        print("No model to preload, starting from scratch")
        print("Generating results.json file")
        with open(get_results_path(config), "w") as f:
            results = {"loss": [], "val_loss": []}
            f.write(json.dumps(results))

    loss_fn = nn.MSELoss().to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        losses = []
        for batch in batch_iterator:

            encoder_input = batch["encoder_input"].to(device) # (b, seq_len) b = Batch
            decoder_input = batch["decoder_input"].to(device) # (b, seq_len)
            encoder_mask = None # default for bidirectional encoding 
            decoder_mask = batch["decoder_mask"].to(device) # (b, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (b, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (b, seq_len, d_model)
            proj_output = model.project(decoder_output) # (b, seq_len, y_size)

            # Compare the output with the label
            label = batch["label"].to(device) # (b, seq_len)

            # Compute the loss using MSE
            #loss = loss_fn(proj_output.view(-1, config["y_size"]), label.view(-1))
            loss = loss_fn(proj_output, label)
            losses.append(loss.item())
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)

            global_step += 1

        # store the mean loss
        mean_loss = sum(losses)/len(losses)
        print(results)
        results["loss"].append(mean_loss)

        # run validation
        results = run_validation(model, val_dataloader, config["tgt_len"], device, "Validation", global_step, writer, epoch, results) # run validation
        
        # save the results
        with open(get_results_path(config), "w") as f:
            f.write(json.dumps(results))

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
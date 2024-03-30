import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from model import build_transformer
from dataset_utils import generate_dataset
from dataset import SequencesDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weight_file_path

from torch.utils.tensorboard import SummaryWriter

import warnings
from pathlib import Path
from tqdm import tqdm
import os

def inference_decoding(model, source, source_mask, seq_len, device):

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input
    decoder_input = torch.empty(1, 1).type_as(source).to(device)
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


def run_validation(model, validation_ds, seq_len, device, print_msg, global_step, writer):
    
    model.eval()

    src_seqs = []
    expected = []
    predicted = []

    try:
        with os.popen("stty size", "r") as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = None # default for bidirectional encoding
            
            model_out = inference_decoding(model, encoder_input, encoder_mask, seq_len, device)

            src_seqs = batch["encoder_input"]
            tgt_seqs = batch["label"]
            pred_seqs = model_out.detach().cpu().numpy()
            for src_seq, tgt_seq, pred_seq in zip(src_seqs, tgt_seqs, pred_seqs):
                src_seqs.append(src_seq)
                expected.append(tgt_seq)
                predicted.append(pred_seq)
        

        # Evaluate the model predictions
        if writer:
            mse_loss = nn.MSELoss()
            mse = mse_loss(torch.tensor(expected), torch.tensor(predicted))
            writer.add_scalar(f"{print_msg} MSE", mse, global_step)
            writer.flush()

def get_ds(config):

    X_raw, y_raw = generate_dataset(config["security"], config["provider"], config["src_len"], config["tgt_len"]) # generate raw dataset

    full_ds = SequencesDataset(X_raw, y_raw, config["src_len"], config["tgt_len"]) # generate dataset object

    # 80% training, 10% validation, 10% test
    train_ds_size = int(0.8 * len(full_ds))
    test_ds_size = int(0.1 * len(full_ds))
    val_ds_size = len(full_ds) - train_ds_size - test_ds_size 

    train_ds, test_ds, val_ds = random_split(full_ds, [train_ds_size, test_ds_size, val_ds_size], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)
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
    writer = SummaryWriter(config["experiment_name"])

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
    else:
        print("No model to preload, starting from scratch")

    loss_fn = nn.MSELoss().to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
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
            loss = loss_fn(proj_output.view(-1, config["y_size"]), label.view(-1))
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

        # run validation
        run_validation(model, val_dataloader, config["tgt_len"], device, "Validation", global_step, writer) # run validation

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


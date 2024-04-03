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
import pprint 


def inference_decoding(model, source, source_mask, seq_len, device, config):

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)  #(Batch, src_len, d_model)
    # Initialize the decoder input
    decoder_input = torch.zeros(encoder_output.shape[0], 1, config["y_size"]).type_as(source).to(device)
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
    
    return decoder_input


def run_validation(model, validation_ds, seq_len, device, print_msg, global_step, writer, epoch, results):
    
    model.eval()

    losses = []
    compounded_returns_losses = []

    pred_classes = [] # these contain data for the cumulative accuracy classification metric
    actual_classes = []

    # !!! Evaluate wheter to keep or remove this try/except block !!! 
    try:
        with os.popen("stty size", "r") as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80
    # !!! Evaluate wheter to keep or remove this try/except block !!! 

    batch_iterator = tqdm(validation_ds, desc=f"Processing Validation for Epoch {epoch:02d}")
    with torch.no_grad():
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = None # default for bidirectional encoding            
            model_out = inference_decoding(model, encoder_input, encoder_mask, seq_len, device, config)

            tgt_seqs = batch["label"].to(device) 

            # Evaluate the model predictions
            mse_loss = nn.MSELoss()
            mse = mse_loss(torch.tensor(tgt_seqs), torch.tensor(model_out))
            losses.append(mse)

            # Extract compounded returns for model_out and label
            proj_output_compound, label_compound = calculate_batch_cumulative_returns(model_out, tgt_seqs)
            compounded_returns_loss = mse_loss(proj_output_compound, label_compound)
            compounded_returns_losses.append(compounded_returns_loss)
            batch_iterator.set_postfix({"val loss": f"{mse.item():6.3f}", "val C_loss": f"{compounded_returns_loss.item():6.3f}"})

            # append batch accuracy on returns
            tensor_pred_classes, tensor_actual_classes = classify_cumulative_returns(proj_output_compound, label_compound)
            pred_classes.extend(tensor_pred_classes.tolist())
            actual_classes.extend(tensor_actual_classes.tolist())

        # calculates and logs mean val loss
        mean_val_loss = sum(losses)/len(losses)
       
        # store the mean val loss into results
        results["val_loss"].append(mean_val_loss.item())

        # calculates and logs mean val C_loss
        mean_val_c_loss = sum(compounded_returns_losses)/len(compounded_returns_losses)
        
        # store the mean val C_loss
        results["val_C_loss"].append(mean_val_c_loss.item())

        # Calculate the overall accuracy on returns
        pred_classes = torch.tensor(pred_classes)
        actual_classes = torch.tensor(actual_classes)
        accuracy = (pred_classes == actual_classes).sum() / pred_classes.shape[0]
        results["val_cumulative_accuracy"].append(accuracy.item())
      
        # Printing epoch metrics to the console
        print(f"VALIDATION RESULTS: Epoch {epoch} - Loss: {mean_val_loss:6.3f} - C_Loss: {mean_val_c_loss:6.3f} - Binary classification accuracy: {accuracy.item():6.3f}\n")

    
        # !!! This does not account for further metrics updates !!!
        writer.add_scalar(f"{print_msg} MSE", mean_val_loss, global_step)
        writer.flush()
        # !!! This does not account for further metrics updates !!!

    return results


def evaluate_inference(model, test_dataloader, seq_len, device, results):
    model.eval()

    losses = []
    compounded_returns_losses = []

    batch_iterator = tqdm(test_dataloader, desc=f"Running inference")
    pred_classes = []
    actual_classes = []
    
    with torch.no_grad():
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = None # default for bidirectional encoding            
            model_out = inference_decoding(model, encoder_input, encoder_mask, seq_len, device, config)

            tgt_seqs = batch["label"].to(device) 

            # Evaluate the model predictions
            mse_loss = nn.MSELoss()
            mse = mse_loss(torch.tensor(tgt_seqs), torch.tensor(model_out))
            losses.append(mse)

            # Extract compounded returns for model_out and label
            proj_output_compound, label_compound = calculate_batch_cumulative_returns(model_out, tgt_seqs)
            compounded_returns_loss = mse_loss(proj_output_compound, label_compound)
            compounded_returns_losses.append(compounded_returns_loss)
            batch_iterator.set_postfix({"test loss": f"{mse.item():6.3f}", "test C_loss": f"{compounded_returns_loss.item():6.3f}"})

            # append batch accuracy on returns
            tensor_pred_classes, tensor_actual_classes = classify_cumulative_returns(proj_output_compound, label_compound)
            pred_classes.extend(tensor_pred_classes.tolist())
            actual_classes.extend(tensor_actual_classes.tolist())


        # calculates mean test loss
        mean_test_loss = sum(losses)/len(losses)
                
        # store the mean test loss into results
        results["test_loss"].append(mean_test_loss.item())

        # calculates mean test C_loss
        mean_test_c_loss = sum(compounded_returns_losses)/len(compounded_returns_losses)

        # store the mean test C_loss
        results["test_C_loss"] = mean_test_c_loss.item()

        # Calculate the overall accuracy on returns
        pred_classes = torch.tensor(pred_classes)
        actual_classes = torch.tensor(actual_classes)
        accuracy = (pred_classes == actual_classes).sum() / pred_classes.shape[0]
        results["test_cumulative_accuracy"] = accuracy.item()

        # Printing Test result metrics to the console
        print(f"\nMean test loss: {mean_test_loss.item():6.3f}")
        print(f"Mean test C_loss: {mean_test_c_loss.item():6.3f}")
        print(f"Test accuracy on binary classification of cumulative returns: {accuracy.item():6.3f}\n")

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
    
    # Logging configs to the console
    print("\n")
    pp.pprint(config)
    print("\n")

    model = get_model(config).to(device)
    print(f"Number of trainable params: {get_n_trainable_params(model)}")
    print("\n")

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # Get dataset
    train_dataloader, test_dataloader, val_dataloader = get_ds(config)
    
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
            results = {"loss": [], "C_loss": [], "cumulative_accuracy": [], "val_loss": [], "val_C_loss": [], "val_cumulative_accuracy": [], "test_loss": None, "test_C_loss": None, "test_cumulative_accuracy": None}
            f.write(json.dumps(results))

    loss_fn = nn.MSELoss().to(device)

    print("\n")
    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Training Epoch {epoch:02d}")
        losses = []
        compounded_returns_losses = []
        pred_classes = [] # these contain data for the cumulative accuracy classification metric
        actual_classes = []
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
            label = batch["label"].to(device) # (b, seq_len, y_size)

            # Extract compounded returns for proj_output and label
            proj_output_compound, label_compound = calculate_batch_cumulative_returns(proj_output, label)
            compounded_returns_loss = loss_fn(proj_output_compound, label_compound)
            compounded_returns_losses.append(compounded_returns_loss.item())

            # append batch accuracy over compounded returns
            tensor_pred_classes, tensor_actual_classes = classify_cumulative_returns(proj_output_compound, label_compound)
            pred_classes.extend(tensor_pred_classes.tolist())
            actual_classes.extend(tensor_actual_classes.tolist())

            # Compute the loss using MSE
            #loss = loss_fn(proj_output.view(-1, config["y_size"]), label.view(-1))
            loss = loss_fn(proj_output, label)
            losses.append(loss.item())
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}", "C_loss": f"{compounded_returns_loss.item():6.3f}"})

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
        results["loss"].append(mean_loss)

        # store the mean c-loss
        mean_c_loss = sum(compounded_returns_losses)/len(compounded_returns_losses)
        results["C_loss"].append(mean_c_loss)

        # Calculate the overall accuracy on returns
        pred_classes = torch.tensor(pred_classes)
        actual_classes = torch.tensor(actual_classes)
        accuracy = (pred_classes == actual_classes).sum() / pred_classes.shape[0]
        results["cumulative_accuracy"].append(accuracy.item())
        
        # Printing epoch metrics to the console
        print(f"TRAIN RESULTS: Epoch {epoch} - Loss: {mean_loss:6.3f} - C_Loss: {mean_c_loss:6.3f} - Binary classification accuracy: {accuracy.item():6.3f}\n")
        
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

    if config["do_test"]:
        results = evaluate_inference(model, test_dataloader, config["tgt_len"], device, results)
        # save the results
        with open(get_results_path(config), "w") as f:
            f.write(json.dumps(results))


def calculate_batch_cumulative_returns(proj_output, label):

    adjusted_predicted_returns = 1 + proj_output[:, :, 0] # for now we assume close is in 0 dimension
    adjusted_actual_returns = 1 + label[:, :, 0]

    cumulative_predicted_compound_returns = torch.cumprod(adjusted_predicted_returns, dim = -1)
    cumulative_actual_compound_returns = torch.cumprod(adjusted_actual_returns, dim = -1)

    cumulative_predicted_compound_returns -= 1
    cumulative_actual_compound_returns -= 1

    return cumulative_predicted_compound_returns, cumulative_actual_compound_returns 


def classify_cumulative_returns(cumulative_predicted_compound_returns, cumulative_actual_compound_returns):

    """Works with batches"""
    
    cumulative_predicted_compound_returns = cumulative_predicted_compound_returns[:, -1]
    cumulative_actual_compound_returns = cumulative_actual_compound_returns[:, -1]
    pred_classes = cumulative_predicted_compound_returns >= 0
    actual_classes = cumulative_actual_compound_returns >= 0
    
    return pred_classes, actual_classes


def get_n_trainable_params(model):
    """
    Calculates the total number of trainable (requires_grad=True) parameters in a PyTorch model.
    
    Args:
    - model (torch.nn.Module): The model to evaluate.
    
    Returns:
    - int: The total number of trainable parameters.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=1, sort_dicts=False)
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
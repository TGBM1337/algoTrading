from pathlib import Path

def get_config():

    config = {
        "security": "SPX500_USD",
        "provider": "oanda",
        "start_year": 2000,
        "end_year": 2024,
        "train_batch_size": 128,
        "val_batch_size": 128,
        "num_epochs": 10,
        "lr": 1e-3,
        "src_len": 256,
        "tgt_len": 32,
        "input_features": 4,
        "y_size": 4,
        "d_model": 64,
        "N": 6,
        "h": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "model_folder": "checkpoints",
        "model_basename": "algoTransformer",
        "preload": None, # "latest" or epoch or None 
        "experiment_name": "runs/tmodel",
        "datasource": "SPX500_USD",
        "max_train_size": None,
        "max_val_size": None,
        "result_path": "results.json",
        "do_test": True
    }

    return config


def get_weights_file_path(config, epoch: str):

    model_folder = f"{config['security']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}_{epoch}.pt"

    return str(Path(".") / model_folder / model_filename)

def get_results_path(config):
    return f"{config['security']}_{config['model_folder']}/{config['result_path']}"

# Find the latest weights file in the weights folder
def latest_weight_file_path(config):
    model_folder = f"{config['security']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
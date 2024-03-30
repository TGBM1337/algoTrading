from pathlib import Path

def get_config():

    config = {
        "security": "EUR_USD",
        "provider": "oanda",
        "batch_size": 128,
        "num_epochs": 20,
        "lr": 1e-3,
        "src_len": 256,
        "tgt_len": 32,
        "input_features": 10,
        "y_size": 10,
        "d_model": 128,
        "N": 6,
        "h": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "experiment_name": "runs/tmodel",
        "datasource": "EUR_USD"
    }

    return config


def get_weights_file_path(config, epoch: str):

    model_folder = f"{config['security']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"

    return str(Path(".") / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weight_file_path(config):
    model_folder = f"{config['security']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
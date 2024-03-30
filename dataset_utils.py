import pyfinancialdata
import pandas as pd
import numpy as np

def normalize_open_close(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    """Normalize with returns open and close prices. Returns two arrays: open and close returns.
    Opens are normalized based on the preceding close value."""

    opens = data[:, 3]
    closes = data[:, 0]
    assert opens.shape == closes.shape

    normalized_opens = np.zeros_like(opens)
    normalized_closes = (closes - opens) / opens * 100
    normalized_opens[1:] = (opens[1:] - opens[:-1]) / opens[:-1] * 100 # the first value is already zero
    
    return normalized_opens, normalized_closes


def normalize_highs_lows(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    """Normalize highs and lows as returns. Returns the two arrays: highs and lows."""

    opens = data[:, 3]
    highs = data[:, 1]
    lows = data[:, 2]
    assert opens.shape == highs.shape and highs.shape == lows.shape

    normalized_seq_highs = (highs - opens) / opens * 100
    normalized_seq_lows = (lows - opens) / opens * 100
    
    return normalized_seq_highs, normalized_seq_lows


def normalize_timestamp(data: np.ndarray) -> np.ndarray:

    """Normalize dates in a 0 to 1 range."""

    dates = data[:, -1]
    norm_dates = [[date.hour / 24, date.minute / 60, date.weekday() / 6, date.day / 31, date.month / 12] for date in dates]
    return np.asarray(norm_dates)

def generate_dataset(security: str, provider: str, src_len: int, tgt_len:int, first_year: int = 2000, last_year: int = 2024) -> tuple[np.ndarray, np.ndarray]:

    """Dataset generator. This is probably just pure crap that returns X, y"""

    seq_len = src_len + tgt_len
    data = [] # all years dataframes
    for year in range(first_year, last_year + 1):
        try:
            year_data = pyfinancialdata.get(provider, security, year)
            data.append(year_data)
        except KeyError:
            continue
    # columns = [time as index], close, high, low, open, volume
    data = pd.concat(data)
    data.drop("price", axis = 1, inplace=True) # price = close
    data["datetime"] = data.index # add the index as column
    data = data.to_numpy()
    # columns = close, high, low, open, volume, time
    # last_index = data.shape[0] // seq_len * seq_len
    # data = data[:last_index]
    # data_splits = np.array_split(data, data.shape[0] / seq_len)
    # data_splits = np.asarray(data_splits)
    data_splits = np.asarray(np.array_split(data[:data.shape[0] // seq_len * seq_len], data[:data.shape[0] // seq_len * seq_len].shape[0] / seq_len)) # splitting into sequences of seq len

    norm_dates = []
    for i in range(data_splits.shape[0]):
        # normalizing volumes
        data_splits[i, :, 4] = (data_splits[i, :, 4] - np.mean(data_splits[i, :, 4])) / np.std(data_splits[i, :, 4])
        # normalize prices time series
        data_splits[i, :, 1], data_splits[i, :, 2] = normalize_highs_lows(data_splits[i, :, :]) # highs and lows
        data_splits[i, :, 3], data_splits[i, :, 0] = normalize_open_close(data_splits[i, :, :]) # open and close 
        norm_dates.append(normalize_timestamp(data_splits[i, :, :])) # timestamp

    # seq dimensions = (close, high, low, open, volume, hour, minute, day_name_number, day, month) 
    data_splits = np.concatenate((data_splits[:, :, :-1], np.asarray(norm_dates)), axis = 2)

    X = data_splits[:, :src_len, :]
    y = data_splits[:, -tgt_len:, :]

    return X, y
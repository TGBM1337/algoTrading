import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SequencesDataset(Dataset):
    def __init__(self, X, y, src_seq_len, tgt_seq_len) -> None:
        """Initialize the dataset object in a pytorch fashon. Takes X and y"""
        super().__init__()
        self.X = X
        self.y = y
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        X_sample = torch.tensor(self.X[index], dtype = torch.float) # torch.float is 32 bit precision!!! Consider tweeking this for better performance/memory handling
        y_sample = torch.tensor(self.y[index], dtype = torch.float) # (sequence_length, num_features)

        # We assume that only y_sample needs casual masking since it goes inside the decoder
        y_causal_mask = causal_mask(y_sample.size(0))

        label = y_sample # consider keeping only some dimensions to predict specific values (prices, volumes)

        # Double check the size of the tensors to make sure they are all seq_len long
        assert X_sample.size(0) == self.src_seq_len
        assert y_sample.size(0) == self.tgt_seq_len

        return {
            "encoder_input": X_sample, # (src_seq_len)
            "decoder_input": y_sample, # (tgt_seq_len)
            # we assume no encoder mask
            "decoder_mask": y_causal_mask, # (1, tgt_seq_len, tgt_seq_len)
            "label": label, # len depends on the number of dimensions we want to predict (consider create a paramenter for this)
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=0).type(torch.int)
    return mask == 0

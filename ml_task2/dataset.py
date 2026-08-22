from torch.utils.data import Dataset
import torch

class FinanceDataset(Dataset):
    def __init__(self, df, window_size, feature_cols):
        self.data = df[feature_cols].values
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size + 1
        
    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.window_size]
        x = torch.tensor(seq[:-1], dtype=torch.float32)  # все кроме последнего
        y = torch.tensor(seq[-1], dtype=torch.float32)   # последний элемент (цель)
        return x, y
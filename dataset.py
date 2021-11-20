import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import LightningDataModule


class GECDataset(Dataset):
    
    def __init__(self, input_ids, attention_mask, labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx])
        attention_mask = torch.tensor(self.attention_mask[idx])

        if self.labels is not None:
            labels = torch.tensor(self.labels[idx], dtype=torch.long)
            return input_ids, attention_mask, labels
        else:
            return input_ids, attention_mask
        

class GECDataModule(LightningDataModule):

    def __init__(self, train_df, val_df, config):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.config = config
    
    def setup(self, stage=None):
        self.train_dataset = GECDataset(
            self.train_df.input_ids.values,
            self.train_df.attention_mask.values,
            self.train_df.label.values, 
        )
        self.val_dataset = GECDataset(
            self.val_df.input_ids.values,
            self.val_df.attention_mask.values,
            self.val_df.label.values, 
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["train_batch_size"],
            num_workers=self.config["num_workers"], 
            shuffle=True,
            pin_memory=False, 
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["val_batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=True,
            pin_memory=False,
        )

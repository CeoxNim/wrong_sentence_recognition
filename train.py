import gc
import os
from pytorch_lightning.accelerators import accelerator
import yaml
import pickle
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, early_stopping
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer

from transformers import BertTokenizer

from dataset import GECDataModule
from model import BertModel

gc.enable()
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 4, 5"

INPUT_DIR = "./data"
OUTPUT_DIR = "./output"
MODEL_DIR = "./models"

def parse_args():
    parser = argparse.ArgumentParser("Train")
    parser.add_argument("--config", type=str, default="basic.yaml")
    return parser.parse_args()

def load_yaml(filepath):
    with open(filepath, "r") as istream:
        data = yaml.safe_load(istream)
    return data

if __name__ == "__main__":

    args = parse_args()
    config = load_yaml(args.config)

    with open(os.path.join(INPUT_DIR, "train2.pkl"), "rb") as istream:
        df = pickle.load(istream)

    # Create stratified (on target) folds for the training data.
    # shuffle=Flase is better because notes are not similar and are in pairs
    skf = StratifiedKFold(n_splits=config["n_splits"], shuffle=False, random_state=config["seed"])
    for f, (t_, v_) in enumerate(skf.split(df, df["label"])):
        df.loc[v_, "fold"] = f
    df["fold"] = df["fold"].astype(int)
    df["label"] = df["label"].astype(int)
    df.reset_index(drop=True, inplace=True)

    # Train x folds
    for fold in range(config["n_splits"]):
        print(f"*** fold {fold} ***")
        
        train_df = df.loc[df["fold"] != fold]
        val_df = df.loc[df["fold"] == fold]

        dm = GECDataModule(train_df, val_df, config)

        checkpoint_callback = ModelCheckpoint(
            monitor="val_score",
            dirpath=OUTPUT_DIR,
            mode="max",
            filename=f"{config['model_name']}-f{fold}-{{val_score:.4f}}",
            save_top_k=1,
            save_last=False,
        )

        lr_monitor = LearningRateMonitor()
        earlystopping = EarlyStopping(monitor="val_score")

        trainer = Trainer(
            gpus=config["gpus"],
            accelerator=config["accelerator"],
            max_epochs=config["epochs"],
            precision=config["precision"],
            num_sanity_val_steps=0,
            callbacks=[checkpoint_callback, lr_monitor, earlystopping],
        )

        model = BertModel(config)
        trainer.fit(model, datamodule=dm)

        del model
        gc.collect()
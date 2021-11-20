import os
import gc
import yaml
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizer
from model import BertModel
from dataset import GECDataset
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

gc.enable()

def read_from_test(filepath):
    df = pd.read_table(filepath, sep='\t', header=None, engine="python", quoting=3)
    labels = [int(astr[:1]) for astr in list(df[0])]
    notes = [astr[2:] for astr in list(df[0])]
    df["note"] = notes
    df["label"] = labels
    df = df[["note", "label"]] 
    # token and encoding
    tokenizer = BertTokenizer.from_pretrained(config["model_name"])   
    df_note = list(df["note"])
    encoding = tokenizer(df_note, truncation=True, padding=True, max_length=128)

    # add columns to dataframe
    df["input_ids"] = encoding["input_ids"]
    df["attention_mask"] = encoding["attention_mask"]
    df["token_type_ids"] = encoding["token_type_ids"]

    return df

def parse_args():
    parser = argparse.ArgumentParser("Test")
    parser.add_argument("--config", type=str, default="basic.yaml")
    return parser.parse_args()

def load_yaml(filepath):
    with open(filepath, "r") as istream:
        data = yaml.safe_load(istream)
    return data

INPUT_DIR = "./data"
MODEL_DIR = "./models2"

if __name__ == "__main__":
    args = parse_args()
    config = load_yaml(args.config)

    # read data from file
    df = read_from_test(os.path.join(INPUT_DIR, "test.txt"))
    print(df.head())
    
    # save the package
    with open(os.path.join(INPUT_DIR, "test.pkl"), "wb") as ostream:
        pickle.dump(df, ostream)
    
    # predict
    all_predictions = []
    for path in glob(os.path.join(MODEL_DIR, "*.ckpt")):
        print(path)
        model = BertModel(config)
        model.load_state_dict(torch.load(path)["state_dict"])
        model.cuda(1).eval()

        data_set = GECDataset(
            df["input_ids"].values, df["attention_mask"].values, df["label"].values
        )
        data_loader = DataLoader(
            data_set,
            batch_size=config["test_batch_size"],
            num_workers=config["num_workers"],
            shuffle=False,
            pin_memory=False,
        )

        preds = []
        for batch in tqdm(data_loader):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.cuda(1)
            attention_mask = attention_mask.cuda(1)
            labels = labels.cuda(1)

            outputs = model(input_ids, attention_mask, labels)

            logits = torch.max(outputs[1], dim=1)[1]
            logits = logits.detach().cpu().numpy()
            preds = np.concatenate([preds, logits], axis=0)
        all_predictions.append(preds)
        del model
        gc.collect()
    
    all_predictions = np.mean(all_predictions, axis=0)
    all_predictions = [round(x) for x in all_predictions]
    labels = list(df["label"])

    f1score = f1_score(y_true=labels, y_pred=all_predictions)
    accscore = accuracy_score(y_true=labels, y_pred=all_predictions)

    print("accusary_score: ", accscore)
    print("f1_score: ", f1score)

    # save the result
    with open("./output/result2.txt", "w") as ostream:
        for label in labels:
            ostream.write(str(label) + "\n")




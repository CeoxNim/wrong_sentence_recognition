import os
import yaml
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import BertTokenizer

def read_from_train(filepath, label):
    df = pd.read_table(filepath, sep='\t', header=None, engine="python", quoting=3)
    df.columns = ["note"]
    df["label"] = label
    return df

def parse_args():
    parser = argparse.ArgumentParser("Train")
    parser.add_argument("--config", type=str, default="basic.yaml")
    return parser.parse_args()

def load_yaml(filepath):
    with open(filepath, "r") as istream:
        data = yaml.safe_load(istream)
    return data

INPUT_DIR = "./data"

if __name__ == "__main__":
    args = parse_args()
    config = load_yaml(args.config)

    # read data from file
    df_true = read_from_train(os.path.join(INPUT_DIR, "train.cor"), label=1)
    df_false = read_from_train(os.path.join(INPUT_DIR, "train.err"), label=0)
    
    df = pd.DataFrame(index=["note", "label"])
    for i in tqdm(range(len(df_true))):
        df = df.append(df_true.iloc[i])
        df = df.append(df_false.iloc[i])

    df.reset_index(drop=True, inplace=True)
    print(df.head())

    # token and encoding
    tokenizer = BertTokenizer.from_pretrained(config["model_name"])   
    df_note = list(df["note"])
    encoding = tokenizer(df_note, truncation=True, padding=True, max_length=128)

    # add columns to dataframe
    df["input_ids"] = encoding["input_ids"]
    df["attention_mask"] = encoding["attention_mask"]
    df["token_type_ids"] = encoding["token_type_ids"]

    # save the package
    with open(os.path.join(INPUT_DIR, "preprocess2.pkl"), "wb") as ostream:
        pickle.dump(df, ostream)
import os
import yaml
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import BertTokenizer

FILE0 = "./data/train.err"
FILE1 = "./data/train.cor"


def parse_args():
    parser = argparse.ArgumentParser("Train")
    parser.add_argument("--config", type=str, default="basic.yaml")
    return parser.parse_args()

def load_yaml(filepath):
    with open(filepath, "r") as istream:
        data = yaml.safe_load(istream)
    return data

def file_union(file0, file1, NUM=590000):
    with open("./data/train.txt", "w") as ostream:
        with open(file0, "r") as f0:
            with open(file1, "r") as f1:
                for i in tqdm(range(NUM)):
                    l0 = f0.readline()
                    l1 = f1.readline()
                    ostream.write("0 " + l0)
                    ostream.write("1 " + l1)

def read_from_train(filepath):
    df = pd.read_table(filepath, sep='\t', header=None, engine="python", quoting=3)
    df.columns = ["note"]
    astrs = list(df["note"])
    notes = [astr[2: ] for astr in astrs]
    labels = [astr[: 1] for astr in astrs]
    df["note"] = notes
    df["label"] = labels
    return df[["note", "label"]]

if __name__ == "__main__":
    args = parse_args()
    config = load_yaml(args.config)

    file_union(FILE0, FILE1)

    df = read_from_train("./data/train.txt")
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
    with open("./data/train2.pkl", "wb") as ostream:
        pickle.dump(df, ostream)




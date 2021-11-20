import os
import yaml
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import BertTokenizer

def read_from_file(filepath, label):
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
OUTPUT_DIR = "./output"

if __name__ == "__main__":
    args = parse_args()
    config = load_yaml(args.config)

    # read data from file
    df_true = read_from_file(os.path.join(INPUT_DIR, "train.cor"), label=1)
    df_false = read_from_file(os.path.join(INPUT_DIR, "train.err"), label=0)
    # df = pd.concat([df_true, df_false])

    df = pd.DataFrame(index=["note", "label"])
    for i in range(len(df_true)):
        df.append(df_true.iloc[i])
        df.append(df_false.iloc[i])

    df.reset_index(drop=True, inplace=True)
    print(df.head())

    # token and encoding
    tokenizer = BertTokenizer.from_pretrained(config["model_name"])   
    df_note = list(df["note"])
    encoding = tokenizer(df_note)

    # visualize the length of the input_ids and save
    input_ids = encoding["input_ids"]
    input_length = [len(i) for i in input_ids]

    sns_density = sns.distplot(input_length)
    density_fig = sns_density.get_figure()
    density_fig.savefig(os.path.join(OUTPUT_DIR, "hist.png"))
    
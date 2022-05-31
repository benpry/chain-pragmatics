"""
This file preprocesses the Katz corpus
"""
import pandas as pd
import numpy as np
from pyprojroot import here
from prompt_generation import make_katz_prompt

train_size = 30
dev_size = 100

if __name__ == "__main__":

    # read in the data
    df_katz = pd.read_csv(here("data/katz-corpus/katz-corpus-full.csv"))

    # exclude the rows we want to exclude
    df_katz = df_katz[df_katz["Include"] == 1]

    # generate the prompts and values
    prompts_with_values = df_katz.apply(lambda x: make_katz_prompt(x), axis=1)
    prompts = [x[0] for x in prompts_with_values]
    values = [x[1] for x in prompts_with_values]
    df_katz["prompt"], df_katz["values"] = prompts, values

    print(df_katz["prompt"][0])

    # split into train, dev, and test
    df_katz = df_katz.sample(frac=1)  # shuffle the data
    # create train, dev, and test data
    df_katz_train = df_katz[:train_size]
    df_katz_dev = df_katz[train_size:train_size + dev_size]
    df_katz_test = df_katz[train_size + dev_size:]

    # save the data
    df_katz_train.to_csv(here("data/katz-corpus/katz-corpus-train.csv"), index=False)
    df_katz_dev.to_csv(here("data/katz-corpus/katz-corpus-dev.csv"), index=False)
    df_katz_test.to_csv(here("data/katz-corpus/katz-corpus-test.csv"), index=False)

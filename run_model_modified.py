# %%
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import circuitsvis as cv
import einops
import numpy as np
import torch as t
from IPython.display import display
from jaxtyping import Float
from nnsight import CONFIG, LanguageModel
from openai import OpenAI
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from datasets import load_dataset

import pickle

logging.disable(sys.maxsize)

t.set_grad_enabled(False)
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# %%
n_new_tokens = 600
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
CONFIG.set_default_api_key("KEY")
model = LanguageModel(model_name, device_map="auto", torch_dtype=t.bfloat16)

def chunk_list(lst, chunk_size=100):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def worker(out_string_modified_list, subject_choice, lag, rep):
    print(f"Running Subject: {subject_choice}, Lag: {lag}, Rep: {rep} ...")
    
    out_dir = Path(subject_choice)
    pkl_file = out_dir / f"tokens_lag{lag}_rep{rep}.pkl"
    if pkl_file.exists():
        print(f"File {pkl_file} already exists. Skipping...")
        return
    chunks = chunk_list(out_string_modified_list, 70)
    tokens = []
    for i_prompts in chunks:
        with model.generate(remote=True, max_new_tokens=n_new_tokens) as generator:
            with generator.invoke(i_prompts):
                i_tokens = model.generator.output.save()
        tokens += i_tokens.value.squeeze().tolist()

    with open(pkl_file, "wb") as f:
        pickle.dump(tokens, f)

sub_list = ['college_medicine']
# sub_list = ['college_computer_science']

for subject_choice in sub_list:
    out_dir = Path(subject_choice)
    for lag in range(1, 4):
        with open(out_dir / f"out_string_modified_lag{lag}.pkl", "rb") as f:
            out_string_modified_list = pickle.load(f)

        for rep in range(5):
            worker(out_string_modified_list, subject_choice, lag, rep)

# %%

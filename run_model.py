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
from multiprocessing import Process


# %%
logging.disable(sys.maxsize)

t.set_grad_enabled(False)
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# %% Load Model
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
model = LanguageModel(model_name, device_map="auto", torch_dtype=t.bfloat16)
CONFIG.set_default_api_key("KEY")

tokenizer = model.tokenizer

N_HEADS = model.config.num_attention_heads
N_LAYERS = model.config.num_hidden_layers
D_MODEL = model.config.hidden_size
D_HEAD = D_MODEL // N_HEADS

print(f"Number of heads: {N_HEADS}")
print(f"Number of layers: {N_LAYERS}")
print(f"Model dimension: {D_MODEL}")
print(f"Head dimension: {D_HEAD}\n")

# print("Entire config: ", model.config)

# %% Load Dataset
# ds = load_dataset("gsm8k", "main")
ds = load_dataset("cais/mmlu", "all")
test_data = ds["test"]

## %%
prompt_template = '''Human: {question}

Choices:
(A) {choices[0]}
(B) {choices[1]}
(C) {choices[2]}
(D) {choices[3]}

Think step by step but be brief. Answer with the letter of the correct choice.
Assistant: '''

# %%
choice_list = ["(A)", "(B)", "(C)", "(D)"]
str_answer_list = ["A", "B", "C", "D"]
token_answer_list = tokenizer.convert_tokens_to_ids(["A", "B", "C", "D", "ĠA", "ĠB", "ĠC", "ĠD"])
THINKING_END_TOKEN = tokenizer.convert_tokens_to_ids("</think>")
BOS_TOKEN = tokenizer.convert_tokens_to_ids("<｜begin▁of▁sentence｜>")
EOS_TOKEN = tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")

def analyze_answer(out_string):
    out_final_answer = None
    if "Answer" in out_string:
        out_final_answer_string = out_string.split("Answer")[-1]
    elif "answer" in out_string:
        out_final_answer_string = out_string.split("answer")[-1]
    else:
        out_final_answer_string = "".join(out_string.split("\n")[-3:])
    
    for i_str in out_final_answer_string[::-1]:
        if i_str in str_answer_list:
            indices = [j for j, c in enumerate(str_answer_list) if c == i_str][0]
            # print(f"Token: {token}, Indices: {indices}")
            out_final_answer = choice_list[indices]
            break
    
    return out_final_answer

class DatasetItem:
    def __init__(self, test_data_i):
        self.data = test_data_i
        # self.in_prompt = None
        self.subject = test_data_i["subject"]
        self.is_finished = False
        self.is_thinking = False
        self.out_token_all = None
        self.out_token = None
        self.out_token_thinking = None
        self.out_decoded = None
        self.out_thinking = None
        self.out_thinking_only = None
        self.out_answer_only = None
        self.out_final_answer = None
    
    @property
    def prompt(self):
        return prompt_template.format(
            question=self.data["question"],
            choices=self.data["choices"]
        )

    @property
    def correct_answer(self):
        return choice_list[self.data["answer"]]

    @property
    def is_correct(self):
        return self.out_final_answer == self.correct_answer

    
    def set_out_token(self, out_token):
        self.out_token_all = out_token
        if out_token[-1] != EOS_TOKEN:
            self.is_finished = False
        else:
            self.is_finished = True

        out_token = self._remove_pre_ext_token(out_token)
        out_token = self._remove_post_ext_token(out_token)
        self.out_token = out_token

        decoded = tokenizer.decode(out_token)
        # decoded = decoded.split("<｜begin▁of▁sentence｜>")[1]
        # decoded = decoded.split("<｜end▁of▁sentence｜>")[0]
        self.out_decoded = decoded
        out_only = decoded.split("Assistant: ")[1]
        
        if "</think>" in decoded:            
            self.is_thinking = True
            self.out_thinking = decoded.split("</think>")[0] + "</think>"
            self.out_thinking_only = out_only.split("</think>")[0] + "</think>"
            self.out_answer_only = out_only.split("</think>")[1]

            for i, i_token in enumerate(out_token):
                if i_token == THINKING_END_TOKEN:
                    self.out_token_thinking = out_token[:i+1]
                    break
        else:
            self.out_answer_only = out_only
        
        if not self.is_finished:
            self.out_final_answer = None
        else:
            self.out_final_answer = analyze_answer(decoded)
            # self._analyze_answer()
            # self.out_final_answer = [a.split("</think>")[0] for a in self.out_answer_only]
    
    def _remove_pre_ext_token(self, token):
        for i, x in enumerate(token):
            if x != EOS_TOKEN and x != BOS_TOKEN:
                return token[i:]

    def _remove_post_ext_token(self, token):
        for i, x in enumerate(token[::-1]):
            if x != EOS_TOKEN:
                return token[:len(token)-i]

# %%
subject_choice_dict = {}
for subject_choice in set(test_data["subject"]):
    # if "college" in subject_choice or "high_school" in subject_choice:
    if "college" in subject_choice:
        sub_nums = sum(np.array(test_data["subject"]) == subject_choice)
        subject_choice_dict[subject_choice] = sub_nums
        print(f"Subject: {subject_choice}, Number of questions: {sub_nums}")

subject_choice_list = sorted(subject_choice_dict.items(), key=lambda x: (x[1], x[0]), reverse=False)
subject_choice_dict = dict(subject_choice_list)

# %%
def chunk_list(lst, chunk_size=100):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

# %% org model run and save
# n_new_tokens = 1000

# for subject_choice in subject_choice_dict.keys():
#     print(f"Running Subject: {subject_choice} ...")
#     ques_idx_choice = [i for i, x in enumerate(test_data) if x["subject"] == subject_choice]
#     test_data_choice = [test_data[i] for i in ques_idx_choice]
#     test_data_inst = [DatasetItem(item) for item in test_data_choice]

#     prompts = [data_inst.prompt for data_inst in test_data_inst]
#     chunks = chunk_list(prompts, 100)
#     tokens = []

#     for i_prompts in chunks:
#         with model.generate(remote=True, max_new_tokens=n_new_tokens) as generator:
#             with generator.invoke(i_prompts):
#                 i_tokens = model.generator.output.save()
#         tokens += i_tokens.value.squeeze().tolist()

#     out_dir = Path(subject_choice)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     with open(out_dir / "tokens.pkl", "wb") as f:
#         pickle.dump(tokens, f)

# %% org model load
test_data_inst_dict = {}
for subject_choice in subject_choice_dict.keys():
    # if subject_choice == "college_medicine":
    #     continue

    ques_idx_choice = [i for i, x in enumerate(test_data) if x["subject"] == subject_choice]
    test_data_choice = [test_data[i] for i in ques_idx_choice]

    test_data_inst = [DatasetItem(item) for item in test_data_choice]

    out_dir = Path(subject_choice)
    with open(out_dir / "tokens.pkl", "rb") as f:
        tokens = pickle.load(f)
    
    for i, (token, data_inst) in enumerate(zip(tokens, test_data_inst)):
        data_inst.set_out_token(token)
    
    test_data_inst_dict[subject_choice] = test_data_inst

# %% select_indices save and load
# select_idx_dict = {}
# for subject_choice in subject_choice_dict.keys():
#     select_idx = [data_inst.is_finished and data_inst.is_thinking
#                   for data_inst in test_data_inst_dict[subject_choice]]
#     select_idx = np.where(select_idx)[0]
#     select_idx_dict[subject_choice] = select_idx
#     print(len(select_idx))

# select_idx_dict = {}
# for subject_choice in subject_choice_dict.keys():
#     select_idx = [data_inst.is_finished and data_inst.is_thinking and data_inst.out_final_answer is not None
#                   for data_inst in test_data_inst_dict[subject_choice]]
#     select_idx = np.where(select_idx)[0]
#     select_idx_dict[subject_choice] = select_idx.tolist()
#     print(len(select_idx))

# for subject_choice in subject_choice_dict.keys():
#     select_idx = select_idx_dict[subject_choice].copy()
#     for idx in select_idx:
#         data_inst = test_data_inst_dict[subject_choice][idx]

#         out_token_modified = data_inst.out_token_thinking.copy()
#         prompt_len = len(tokenizer.tokenize(data_inst.prompt))
#         for i, i_token in enumerate(out_token_modified[::-1]):
#             if i_token in token_answer_list:
#                 index = [j for j, c in enumerate(token_answer_list) if c == i_token][0]
#                 break
#         if (len(out_token_modified) - 1 - i) <= prompt_len:
#             select_idx_dict[subject_choice].remove(idx)
#             print("No modification needed")
#             continue

# for subject_choice in subject_choice_dict.keys():
#     out_dir = Path(subject_choice)
#     with open(out_dir / "select_indices.pkl", "wb") as f:
#         pickle.dump(select_idx_dict[subject_choice], f)

select_idx_dict = {}
for subject_choice in subject_choice_dict.keys():
    out_dir = Path(subject_choice)
    with open(out_dir / "select_indices.pkl", "rb") as f:
        select_idx = pickle.load(f)
    select_idx_dict[subject_choice] = select_idx
    print(len(select_idx))
 
# %% modified model prompt and save
n_new_tokens = 600

processes = []
select_idx_list = sorted(select_idx_dict.items(), key=lambda x: (len(x[1]), x[0]), reverse=False)
select_idx_dict = dict(select_idx_list)

for subject_choice, select_idx in select_idx_dict.items():
    out_dir = Path(subject_choice)
    for lag in range(1, 4):
        out_string_modified_list = []
        for idx in select_idx:
            data_inst = test_data_inst_dict[subject_choice][idx]

            out_token_modified = data_inst.out_token_thinking.copy()
            for i, i_token in enumerate(out_token_modified[::-1]):
                if i_token in token_answer_list:
                    index = [j for j, c in enumerate(token_answer_list) if c == i_token][0]
                    break

            out_token_modified[len(out_token_modified) - 1 - i] = token_answer_list[index // 4 * 4 + (index - lag) % 4]

            out_string_modified = tokenizer.decode(out_token_modified)
            out_string_modified_list.append(out_string_modified)

        with open(out_dir / f"out_string_modified_lag{lag}.pkl", "wb") as f:
            pickle.dump(out_string_modified_list, f)

#         for rep in range(5):
#             p = Process(target=worker, args=(out_string_modified_list, subject_choice, lag, rep))
#             p.start()
#             processes.append(p)
#             time.sleep(60)

#             # worker(out_string_modified_list, subject_choice, lag, rep)
            
#             # print(f"Running Subject: {subject_choice}, Lag: {lag}, Rep: {rep} ...")
#             # with model.generate(remote=True, max_new_tokens=n_new_tokens) as generator:
#             #     with generator.invoke(out_string_modified_list):
#             #         tokens = model.generator.output.save()
            
#             # tokens = tokens.value.squeeze().tolist()
#             # with open(out_dir / f"tokens_lag{lag}_rep{rep}.pkl", "wb") as f:
#             #     pickle.dump(tokens, f)

# for p in processes:
#     p.join()

# for i, i_token in enumerate(out_token_modified[::-1]):
#     if i_token in token_answer_list:
#         index = [j for j, c in enumerate(token_answer_list) if c == i_token][0]
#         break

# if (mod_pos := len(out_token_modified) - 1 - i) > prompt_len:
#     out_token_modified[mod_pos] = token_answer_list[index // 4 * 4 + (index - 1) % 4]
# else:
#     print("No modification needed")

# out_string_modified = tokenizer.decode(out_token_modified)

# with model.generate(remote=True, max_new_tokens=600) as generator:
#     with generator.invoke(out_string_modified):
#         tokens = model.generator.output.save()

# select_idx = [data_inst.is_finished and data_inst.is_thinking for data_inst in test_data_inst]
# select_idx = np.where(select_idx)[0]

# %%
# with open("/mnt/c/Users/zhwen/Dropbox/AIsafety/AISES/project/college_chemistry/out_string_modified_lag1.pkl", "rb") as f:
#     out_string_modified_list = pickle.load(f)
# %%

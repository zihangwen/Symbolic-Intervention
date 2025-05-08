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
from ml_dtypes import bfloat16

import pickle
# from plot_utils import imshow

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

# %%
subject_choice = "college_mathematics"
ques_idx_choice = [i for i, x in enumerate(test_data) if x["subject"] == subject_choice]
test_data_choice = [test_data[i] for i in ques_idx_choice]

# %%
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

    # def _analyze_answer(self):
    #     if "Answer" in self.out_answer_only:
    #         self.out_final_answer_string = self.out_answer_only.split("Answer")[-1]
    #     elif "answer" in self.out_answer_only:
    #         self.out_final_answer_string = self.out_answer_only.split("answer")[-1]
    #     else:
    #         self.out_final_answer_string = "".join(self.out_answer_only.split("\n")[-3:])
        
    #     for i_str in self.out_final_answer_string[::-1]:
    #         if i_str in str_answer_list:
    #             indices = [j for j, c in enumerate(str_answer_list) if c == i_str][0]
    #             # print(f"Token: {token}, Indices: {indices}")
    #             self.out_final_answer = choice_list[indices]
    #             return
            
        # for token in self.out_token[::-1]:
        #     if token in token_answer_list:
        #         indices = [j for j, c in enumerate(token_answer_list) if c == token][0]
        #         # print(f"Token: {token}, Indices: {indices}")
        #         self.out_final_answer = choice_list[indices % 4]
        #         return


# %%
test_data_inst = [DatasetItem(item) for item in test_data_choice]

# %% no hint
# prompts = [data_inst.prompt for data_inst in test_data_inst]

# n_new_tokens = 1000
# with model.generate(remote=True, max_new_tokens=n_new_tokens) as generator:
#     with generator.invoke(prompts):
#         tokens = model.generator.output.save()

# with open("tokens.pkl", "wb") as f:
#     pickle.dump(tokens.value.squeeze().tolist(), f)

with open("college_mathematics/tokens.pkl", "rb") as f:
    tokens = pickle.load(f)


# %%
for i, (token, data_inst) in enumerate(zip(tokens, test_data_inst)):
    data_inst.set_out_token(token)

# %% analysis
finish_idx = [data_inst.is_finished and data_inst.is_thinking for data_inst in test_data_inst]
finish_idx = np.where(finish_idx)[0]

# %%
idx = finish_idx[2]

table = Table("Prompt", "Correct answer", "LLM thinking", "LLM answer")
data_inst = test_data_inst[idx]
# table.add_row(data_inst.prompt, data_inst.correct_answer, data_inst.out_thinking_only, data_inst.out_final_answer)
table.add_row(data_inst.prompt, data_inst.correct_answer, (data_inst.out_thinking_only[:628] + "\n\n...\n" + data_inst.out_thinking_only[-910:]).strip(), data_inst.out_answer_only.strip())
rprint(table)

# %% ----- ----- ----- token level modification ----- ----- ----- %% #
out_token_modified = data_inst.out_token_thinking.copy()
prompt_len = len(tokenizer.tokenize(data_inst.prompt))

# token_change_list = tokenizer.convert_tokens_to_ids(["Ġtrue", "Ġfalse"])

# target_idx = [tok in token_change_list for tok in data_inst.out_token_thinking]
# target_idx = np.where(target_idx)[0]
# print(f"number of target tokens: {len(target_idx)}")
# t_idx = target_idx[3]
# for t_idx in target_idx:
#     if t_idx > prompt_len:
#         out_token_modified[t_idx] = token_change_list[1] if out_token_modified[t_idx] == token_change_list[0] else token_change_list[0]

for i, i_token in enumerate(out_token_modified[::-1]):
    if i_token in token_answer_list:
        index = [j for j, c in enumerate(token_answer_list) if c == i_token][0]
        break

if (mod_pos := len(out_token_modified) - 1 - i) > prompt_len:
    out_token_modified[mod_pos] = token_answer_list[index // 4 * 4 + (index - 1) % 4]
else:
    print("No modification needed")

out_string_modified = tokenizer.decode(out_token_modified)

# with model.generate(remote=True, max_new_tokens=600) as generator:
#     with generator.invoke(out_string_modified):
#         tokens1 = model.generator.output.save()

# with open("case_study/tokens_modified_1.pkl", "wb") as f:
#     pickle.dump(tokens1.value.squeeze().tolist(), f)

# with open("case_study/tokens_modified_2.pkl", "wb") as f:
#     pickle.dump(tokens2.value.squeeze().tolist(), f)

# %%
with open("case_study/tokens_modified_1.pkl", "rb") as f:
    tokens1 = pickle.load(f)
with open("case_study/tokens_modified_2.pkl", "rb") as f:
    tokens2 = pickle.load(f)
# with open("case_study/tokens_modified_1.pkl", "wb") as f:
#     pickle.dump(tokens1.value.squeeze().tolist(), f)

# %%
table = Table("Prompt", "LLM thinking", "LLM thinking (intervention)")
data_inst = test_data_inst[idx]
out_string_modified_thinking = out_string_modified.split("Assistant: ")[1]
table.add_row(
    data_inst.prompt,
    (data_inst.out_thinking_only[:31] + "\n\n...\n" + data_inst.out_thinking_only[-38:]).strip(),
    (out_string_modified_thinking[:31] + "\n\n...\n" + "\nSo the false statement is [b green]C[/].\n</think>").strip()
)
rprint(table)

# %%
table = Table("Prompt", "LLM answer (intervention rep1)", "LLM answer (intervention rep2)")
data_inst = test_data_inst[idx]
out_string_modified_thinking = out_string_modified.split("Assistant: ")[1]
table.add_row(
    data_inst.prompt,
    tokenizer.decode(tokens1).split("</think>")[1].strip(),
    tokenizer.decode(tokens2).split("</think>")[1].strip(),
)
rprint(table)

# %%
right_prompt = data_inst.out_decoded[:-67]
wrong_prompt = tokenizer.decode(tokens1[1:])[:-33]

table = Table("Original output", "Intervention output")
table.add_row(right_prompt[:600] + '\n\n...\n\n' + right_prompt[-100:], 
              wrong_prompt[:600] + '\n\n...\n\n' + wrong_prompt[-100:])
rprint(table)

# %%
with model.trace(remote=False) as tracer:
    with tracer.invoke(right_prompt):
        # original_output = model.model.layers[-1].output[0].clone().save()
        original_logits = model.lm_head.output[0, -1].save()
        # original_attn_patterns = model.model.layers[0].self_attn.output[1].save()
        # original_query = model.model.layers[0].self_attn.q_proj.output[0].save()
        # original_key = model.model.layers[0].self_attn.k_proj.output[0].save()

    with tracer.invoke(wrong_prompt):
        # modified_output = model.model.layers[-1].output[0].clone().save()
        modified_logits = model.lm_head.output[0, -1].save()
        # modified_attn_patterns = model.model.layers[0].self_attn.output[1].save()
        # modified_query = model.model.layers[0].self_attn.q_proj.output[0].save()
        # modified_key = model.model.layers[0].self_attn.k_proj.output[0].save()

# with open("case_study/original_output.pkl", "wb") as f:
#     pickle.dump(original_output.value, f)
with open("case_study/original_logits.pkl", "wb") as f:
    pickle.dump(original_logits, f)
# with open("case_study/original_query.pkl", "wb") as f:
#     pickle.dump(original_query.value, f)
# with open("case_study/original_key.pkl", "wb") as f:
#     pickle.dump(original_key.value, f)

# with open("case_study/modified_output.pkl", "wb") as f:
#     pickle.dump(modified_output.value, f)
with open("case_study/modified_logits.pkl", "wb") as f:
    pickle.dump(modified_logits, f)
# with open("case_study/modified_query.pkl", "wb") as f:
#     pickle.dump(modified_query.value, f)
# with open("case_study/modified_key.pkl", "wb") as f:
#     pickle.dump(modified_key.value, f)

# %%
# with open("case_study/original_output.pkl", "rb") as f:
#     original_output = pickle.load(f)
# with open("case_study/original_logits.pkl", "rb") as f:
#     original_logits = pickle.load(f)
# with open("case_study/original_query.pkl", "rb") as f:
#     original_query = pickle.load(f)
# with open("case_study/original_key.pkl", "rb") as f:
#     original_key = pickle.load(f)

# with open("case_study/modified_output.pkl", "rb") as f:
#     modified_output = pickle.load(f)
# with open("case_study/modified_logits.pkl", "rb") as f:
#     modified_logits = pickle.load(f)
# with open("case_study/modified_query.pkl", "rb") as f:
#     modified_query = pickle.load(f)
# with open("case_study/modified_key.pkl", "rb") as f:
#     modified_key = pickle.load(f)

# %%
k = 10
table = Table("Rank", "Token", "Logit", "Token (intervention)", "Logit (intervention)")
topk_original_logits = t.topk(original_logits, k=k, dim=-1)
topk_modified_logits = t.topk(modified_logits, k=k, dim=-1)
for i in range(k):
    token = tokenizer.convert_ids_to_tokens(topk_original_logits.indices[i].item())
    token_mod = tokenizer.convert_ids_to_tokens(topk_modified_logits.indices[i].item())
    token = token.replace('Ġ', ' ')
    token = token.replace('Ċ', '\n')
    token_mod = token_mod.replace('Ġ', ' ')
    token_mod = token_mod.replace('Ċ', '\n')
    table.add_row(
        str(i),
        f"{repr(token)} ({topk_original_logits.indices[i].item()})",
        f"{topk_original_logits.values[i]:.4f}",
        f"{repr(token_mod)} ({topk_modified_logits.indices[i].item()})",
        f"{topk_modified_logits.values[i]:.4f}"
    )
rprint(table)

# for i, (idx, logit) in enumerate(zip(topk_original_logits.indices, topk_original_logits.values)):
#     token = tokenizer.convert_ids_to_tokens(idx.item())
#     print(f"Rank {i}: {token} ({idx.item()})\t\tlogit: {logit:.4f}")

# %%
original_query_mapped = einops.rearrange(original_query, "seq (n_head d_head) -> seq n_head d_head", n_head=N_HEADS)
original_key_mapped = einops.rearrange(original_key, "seq (n_head d_head) -> seq n_head d_head", n_head=model.config.num_key_value_heads)
original_key_mapped = einops.repeat(original_key_mapped, "seq n_head d_head -> seq (repeat n_head) d_head", repeat=8)

modified_query_mapped = einops.rearrange(modified_query, "seq (n_head d_head) -> seq n_head d_head", n_head=N_HEADS)
modified_key_mapped = einops.rearrange(modified_key, "seq (n_head d_head) -> seq n_head d_head", n_head=model.config.num_key_value_heads)
modified_key_mapped = einops.repeat(modified_key_mapped, "seq n_head d_head -> seq (repeat n_head) d_head", repeat=8)

original_attn_patterns = einops.einsum(original_query_mapped, original_key_mapped, "seq_q n_head d_head, seq_k n_head d_head -> n_head seq_q seq_k")
modified_attn_patterns = einops.einsum(modified_query_mapped, modified_key_mapped, "seq_q n_head d_head, seq_k n_head d_head -> n_head seq_q seq_k")

original_attn_patterns = original_attn_patterns[:,1:,1:]
modified_attn_patterns = modified_attn_patterns[:,1:,1:]

original_str_tokens = model.tokenizer.tokenize(right_prompt)
original_str_tokens = [s.replace('Ġ', ' ') for s in original_str_tokens]
original_str_tokens = [s.replace('Ċ', '\n') for s in original_str_tokens]

modified_str_tokens = model.tokenizer.tokenize(wrong_prompt)
modified_str_tokens = [s.replace('Ġ', ' ') for s in modified_str_tokens]
original_str_tokens = [s.replace('Ċ', '\n') for s in original_str_tokens]

# %%
# imshow(
#     original_attn_patterns.mean(0)[800:, 800:].view(dtype=t.uint16).numpy().view(bfloat16),
# )
# imshow(
#     modified_attn_patterns.mean(0)[800:, 800:].view(dtype=t.uint16).numpy().view(bfloat16),
# )
# %%
display(cv.attention.attention_patterns(
    tokens=original_str_tokens[873:],
    attention=original_attn_patterns[:, 873:, 873:],
))

# %%
display(cv.attention.attention_patterns(
    tokens=modified_str_tokens,
    attention=modified_attn_patterns,
))


# %%
# right_prompt = data_inst.out_decoded[:-70]
# wrong_prompt = tokenizer.decode(tokens1.value.squeeze().tolist()[1:])[:-36]

# with model.trace(remote=True) as tracer:
#     with tracer.invoke(right_prompt):
#         original_output2 = model.model.layers[-1].output[0].clone().save()
#         original_logits2 = model.lm_head.output[0, -1].save()
#         original_attn_patterns2  = model.model.layers[0].self_attn.output.save()

#     with tracer.invoke(wrong_prompt):
#         modified_output2 = model.model.layers[-1].output[0].clone().save()
#         modified_logits2 = model.lm_head.output[0, -1].save()
#         modified_attn_patterns2 = model.model.layers[0].self_attn.output.save()


# with model.trace(prompt, remote=REMOTE, scan=True, validate=True):
#     original_output = model.transformer.h[-1].output[0].clone().save()
#     print(f"{model.transformer.h[-1].output.shape=}\n")
#     model.transformer.h[-1].output[0][:, seq_len] = 0
#     modified_output = model.transformer.h[-1].output[0].save()

# print(tokenizer.decode(tokens.value.squeeze().tolist()))
# out_token_inst = [s.replace('Ġ', ' ') for s in out_token_inst]
# out_token_inst = [s.replace('Ċ', '\n') for s in out_token_inst]

# %%
# str_tokens = tokenizer.tokenize(temp_1)
# str_tokens = tokenizer.convert_ids_to_tokens(data_inst.out_token)
# str_tokens = [s.replace('Ġ', ' ') for s in str_tokens]
# str_tokens = [s.replace('Ċ', '\n') for s in str_tokens]
# print("".join(str_tokens))

# %% Display answers:
# table1 = Table("prompt", "answer", "last 10 tokens", "last 10 str tokens", "extracted answer", "idx")
# for idx in finish_idx:
#     data_inst = test_data_inst[idx]
#     last10 = data_inst.out_token[-10:]
#     last10_str = tokenizer.convert_ids_to_tokens(last10)
#     last10_str = [s.replace('Ġ', ' ') for s in last10_str]
#     table1.add_row(data_inst.prompt, data_inst.correct_answer, str(last10),str(last10_str), data_inst.out_final_answer, str(idx))

# rprint(table1)

# %% Display None answers:
# non_ABCD_indices = [idx for idx in finish_idx if test_data_inst[idx].out_final_answer is None]

# table2 = Table("prompt", "answer", "last 10 tokens", "LLM answers", "idx")
# for idx in non_ABCD_indices:
#     data_inst = test_data_inst[idx]
#     last10 = data_inst.out_token[-10:]
#     last10_str = tokenizer.convert_ids_to_tokens(last10)
#     last10_str = [s.replace('Ġ', ' ') for s in last10_str]
#     table2.add_row(data_inst.prompt, data_inst.correct_answer, str(last10), data_inst.out_final_answer_string, str(idx))

# rprint(table2)

# %%
# finished_ABCD_indices = [idx for idx in finish_idx if test_data_inst[idx].out_final_answer is not None]
# table3 = Table("question idx", "prompt", "answer", "LLM answer", "correctness")
# for idx in finished_ABCD_indices:
#     data_inst = test_data_inst[idx]
#     table3.add_row(str(idx), data_inst.prompt, data_inst.correct_answer, data_inst.out_final_answer, str(data_inst.is_correct))

# rprint(table3)

# %%
# for idx in finished_ABCD_indices:
#     data_inst = test_data_inst[idx]
#     print(len(data_inst.out_token) - len(tokenizer.tokenize(data_inst.prompt)))

# %%

import random
import time

import numpy as np
import torch
import tqdm
from transformers import AutoModel, AutoTokenizer

from modeling.custom_attention import Qwen2AttentionCustom, trans
from modeling.modeling_qwen2 import Qwen2Model

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# MODEL_PATH = "/share/project/hcr/models/openai-community/gpt2"
MODEL_PATH = "/share/project/hcr/models/Qwen/Qwen2.5-7B-Instruct"
NUM_PROMPTS = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(custom_attn=False):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Qwen2Model.from_pretrained(MODEL_PATH)

    if custom_attn:
        model = trans(model)
    model.to(DEVICE)
    model.eval()

    text = "Replace me by any text you'd like."
    text *= 64
    encoded_input = tokenizer(text, return_tensors="pt").to(DEVICE)
    return model, encoded_input


def test_once(model, encoded_input, attn_impl=""):
    print("====================================================")
    print(f"Model: {model.config._name_or_path}")
    print(f"AttnImpl: {attn_impl}")

    input_len = len(encoded_input.data["input_ids"][0]) * NUM_PROMPTS
    t_start = time.time()
    output_len = 0
    with torch.no_grad():
        for i in tqdm.trange(NUM_PROMPTS, desc="Processing prompts"):
            output = model(**encoded_input)
            output_len += output.past_key_values[0][0].shape[2]

    time_elapsed = time.time() - t_start
    print(f"Time Elapsed: {time_elapsed}s")
    print(f"Throughput(input):  {input_len/time_elapsed} tokens/s")
    print(f"Throughput(output): {output_len/time_elapsed} tokens/s")
    import pdb

    pdb.set_trace()  # noqa: E999


def test():
    model, encoded_input = init_model(True)
    test_once(model, encoded_input, attn_impl="eager")
    del model, encoded_input

    model, encoded_input = init_model(False)
    test_once(model, encoded_input, attn_impl="custom")
    del model, encoded_input

    model, encoded_input = init_model(True)
    test_once(model, encoded_input, attn_impl="eager")
    del model, encoded_input


if __name__ == "__main__":
    test()

# =========================================
# 8B model - RAW
# Time Elapsed: 11.29983925819397s
# Throughput(input):  5811.056113243769 tokens/s
# Throughput(output): 5811.056113243769 tokens/s

# 8B model - MergeQK

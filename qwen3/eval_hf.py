import evaluate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
MODEL_PATH = "/share/project/hcr/models/Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# 准备评估指标
perplexity = evaluate.load("perplexity")

# 准备评估数据
input_texts = [
    "Hello, how are you today?",
    "What is the meaning of life?",
    "Transformers are revolutionizing NLP.",
]

import pdb

pdb.set_trace()  # noqa: E999
# 计算困惑度
results = perplexity.compute(model_id=MODEL_PATH, texts=input_texts, batch_size=1)

print(f"困惑度: {results['perplexities']}")

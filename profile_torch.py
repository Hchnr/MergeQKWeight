import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, record_function, ProfilerActivity

# 加载模型和分词器
MODEL_PATH = "/share/project/hcr/models/openai-community/gpt2"
MODEL_PATH = "/share/project/hcr/models/Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).cuda()

# 准备输入
input_text = "Hello, how are you today?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()

# 使用 PyTorch 分析器
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    with record_function("model_inference"):
        with torch.no_grad():
            model.generate(input_ids, max_length=50)

# 打印分析结果
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 也可以导出为 HTML 文件进行更详细的分析
prof.export_chrome_trace("trace.json")

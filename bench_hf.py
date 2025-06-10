from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.benchmark import Benchmark, BenchmarkArguments

# 定义基准测试参数
args = BenchmarkArguments(
    models=["/share/project/hcr/models/openai-community/gpt2"],  # 要测试的模型
    batch_sizes=[1, 8],  # 批量大小
    sequence_lengths=[8, 32, 128],  # 序列长度
    # 其他可选参数：frameworks, inference, training 等
)

# 创建基准测试实例
benchmark = Benchmark(args)

# 运行基准测试
results = benchmark.run()
print(results)

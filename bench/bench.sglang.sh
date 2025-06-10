# This script benchmarks the throughput of a model using SGLang's offline throughput benchmark.
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.bench_offline_throughput --model-path /share/project/hcr/models/Qwen/Qwen3-8B --num-prompts 512 --random-input-len 512 --random-output-len 128 --dataset-name random

'''
====== Offline Throughput Benchmark Result =======
Backend:                                 engine
Successful requests:                     512
Benchmark duration (s):                  12.60
Total input tokens:                      130409
Total generated tokens:                  32081
Last generation throughput (tok/s):      6692.56
Request throughput (req/s):              40.62
Input token throughput (tok/s):          10346.51
Output token throughput (tok/s):         2545.27
Total token throughput (tok/s):          12891.78
==================================================
'''
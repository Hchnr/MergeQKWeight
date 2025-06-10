from optimum_benchmark import (
    Benchmark,
    BenchmarkConfig,
    TorchrunConfig,
    InferenceConfig,
    PyTorchConfig,
)
from optimum_benchmark.logging_utils import setup_logging

# setup_logging(level="INFO", handlers=["console"])
setup_logging(level="INFO")
MODEL_PATH = "/share/project/hcr/models/Qwen/Qwen3-8B"


def load_benchmark():
    # load artifacts from the hub
    benchmark = Benchmark.from_hub(
        "IlyasMoutawwakil/pytorch_gpt2"
    )  # or Benchmark.from_hub("IlyasMoutawwakil/pytorch_gpt2")

    # or load them from disk
    benchmark = Benchmark.load_json(
        "benchmark.json"
    )  # or Benchmark.load_csv("benchmark_report.csv")

    return benchmark


if __name__ == "__main__":
    launcher_config = TorchrunConfig(nproc_per_node=1)
    scenario_config = InferenceConfig(latency=True, memory=True)
    backend_config = PyTorchConfig(
        model=MODEL_PATH, device="cuda", device_ids="0", no_weights=True
    )
    benchmark_config = BenchmarkConfig(
        name="qwen3-8b_a800x1",
        scenario=scenario_config,
        launcher=launcher_config,
        backend=backend_config,
    )
    benchmark_report = Benchmark.launch(benchmark_config)

    # convert artifacts to a dictionary or dataframe
    benchmark_config.to_dict()  # or benchmark_config.to_dataframe()

    # save artifacts to disk as json or csv files
    benchmark_report.save_csv(
        "benchmark_report.csv"
    )  # or benchmark_report.save_json("benchmark_report.json")

    # push artifacts to the hub
    # benchmark_config.push_to_hub("IlyasMoutawwakil/pytorch_gpt2")
    # or benchmark_config.push_to_hub("IlyasMoutawwakil/pytorch_gpt2")

    # or merge them into a single artifact
    benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
    benchmark.save_json("benchmark.json")  # or benchmark.save_csv("benchmark.csv")
    # benchmark.push_to_hub("IlyasMoutawwakil/pytorch_gpt2")

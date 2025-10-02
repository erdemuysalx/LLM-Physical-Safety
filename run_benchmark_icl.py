import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from models import LLM
from benchmark import Benchmark

if __name__ == "__main__":
    """
    LLM-Robotics Benchmark with In-Context Learning
    1. Choose model (define a predict function with proper system prompt)
    2. Run benchmark
        a. test with test dataset 
        b. LLM-generated code test on Microsoft AirSim Simulation (only on Windows)
        c. return benchmark result

    model_name:
    "gpt-3.5-turbo", "gpt-4", "gemini-pro", "llama2-7b-chat", "codellama-7b-instruct",
    "mistral-7b-instruct-v0.2", "Meta-Llama-3-8B-Instruct", "codeqwen1.5-7b-chat"
    "codellama-13b-instruct", "codellama-34b-instruct"

    dataset:
    "deliberate", "unintentional", "violation", "utility"
    """
    print("Starting benchmark, please wait...")
    model_name = "gemini-pro"
    datasets = ["deliberate"]
    model = LLM.create_model(model_name=model_name)

    file = open('system_prompts/in_context_learning.txt', 'r')
    icl = file.read()
    file.close()
    print(icl)

    benchmark = Benchmark()
    for dataset in datasets:
        print('')
        benchmark.run(model=model, dataset=dataset, model_name=model_name, icl=icl)

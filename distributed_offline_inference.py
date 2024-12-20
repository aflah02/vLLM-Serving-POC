# Based on - https://docs.vllm.ai/en/stable/getting_started/examples/offline_inference_distributed.html

from typing import Any, Dict, List

import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams
import argparse
import json
import pandas as pd

assert Version(ray.__version__) >= Version(
    "2.22.0"), "Ray version must be at least 2.22.0"

global args

# Create a class to do batch inference.
class LLMPredictor:

    def __init__(self):
        # Create an LLM.
        if args.use_mp_as_distributed_executor_backend:
            self.llm = LLM(model=args.model
                        tensor_parallel_size=args.tp_size,
                            pipeline_parallel_size=args.pp_size,
                                enable_prefix_caching=args.enable_prefix_caching,
                                gpu_memory_utilization=args.gpu_memory_utilization,
                                enforce_eager=args.enforce_eager,
                                distributed_executor_backend="mp",
                                enable_chunked_prefill=args.enable_chunked_prefill
                                )
        else:
            self.llm = LLM(model=args.model
                        tensor_parallel_size=args.tp_size,
                            pipeline_parallel_size=args.pp_size,
                                enable_prefix_caching=args.enable_prefix_caching,
                                gpu_memory_utilization=args.gpu_memory_utilization,
                                enforce_eager=args.enforce_eager,
                                enable_chunked_prefill=args.enable_chunked_prefill
                                )

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(batch["text"], sampling_params)
        prompt: List[str] = []
        generated_text: List[str] = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(' '.join([o.text for o in output.outputs]))
        return {
            "prompt": prompt,
            "generated_text": generated_text,
        }

# For tensor_parallel_size > 1, we need to create placement groups for vLLM
# to use. Every actor has to have its own placement group.
def scheduling_strategy_fn():
    # One bundle per tensor parallel worker
    pg = ray.util.placement_group(
        [{
            "GPU": 1,
            "CPU": 1
        }] * tensor_parallel_size,
        strategy="STRICT_PACK",
    )
    return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
        pg, placement_group_capture_child_tasks=True))

def main(args):
    print("Arguments: ", args)
    
    resources_kwarg: Dict[str, Any] = {}
    if args.tp_size == 1:
        # For tensor_parallel_size == 1, we simply set num_gpus=1.
        resources_kwarg["num_gpus"] = 1
    else:
        # Otherwise, we have to set num_gpus=0 and provide
        # a function that will create a placement group for
        # each instance.
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

    sampling_params = SamplingParams(temperature=0.0, max_tokens=10000)

    # Read Queries 
    df = pd.read_csv(args.queries_path)

    queries = df["Question"].tolist()

    # Read Prompt
    with open(args.prompt_path, "r") as f:
        prompt = f.read()

    conversation = [
        {
            "role": "system",
            "content": prompt
        },
    ]

    ls_conversations = []

    for query in queries:
        conv_temp = conversation.copy()
        conv_temp.append({
            "role": "user",
            "content": query
        })
        ls_conversations.append(conv_temp)

    print("Sample query: ")
    print(ls_conversations[0])

    # Warmup Cache
    print("Warming up cache...")
    llm.chat(ls_conversations[0], sampling_params)

    # Generate completions
    completions = llm.chat(ls_conversations, sampling_params)

    completion_texts = []

    for completion in completions:
        generated_text = completion.outputs[0].text
        completion_texts.append(generated_text)

    # Save completions
    with open(args.save_path, "w") as f:
        json.dump(completion_texts, f)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str)
    argparser.add_argument("--tp_size", type=int)
    argparser.add_argument("--pp_size", type=int)
    argparser.add_argument("--save_path", type=str)
    argparser.add_argument("--queries_path", type=str)
    argparser.add_argument("--prompt_path", type=str)
    argparser.add_argument("--enable_prefix_caching", type=bool, default=True)
    argparser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    argparser.add_argument("--enforce_eager", type=bool, default=False)
    argparser.add_argument("--use-mp-as-distributed_executor_backend", type=bool, default=False)
    argparser.add_argument("--enable_chunked_prefill", type=bool, default=False)
    # Set global args
    args = argparser.parse_args()
    main(args)
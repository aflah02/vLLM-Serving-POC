from vllm import LLM, SamplingParams
import argparse
import json

def main(args):
    print("Arguments: ", args)
    print("Loading model...")
    llm = LLM(args.model, tensor_parallel_size=args.tp_size, pipeline_parallel_size=args.pp_size, enable_prefix_caching=True)
    sampling_params = SamplingParams(temperature=0.0)
    print("Model loaded.")

    # Read Queries (list of strings)
    with open(args.queries_path, "r") as f:
        queries = json.load(f)

    print("Queries: ", queries)

    # Generate completions
    completions = llm.generate(queries, sampling_params)

    print("Completions: ", completions)

    # Save completions
    with open(args.save_path, "w") as f:
        json.dump(completions, f)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default="facebook/opt-125m")
    argparser.add_argument("--tp_size", type=int, default=1)
    argparser.add_argument("--pp_size", type=int, default=1)
    argparser.add_argument("--save_path", type=str)
    argparser.add_argument("--queries_path", type=str)
    args = argparser.parse_args()
    main(args)

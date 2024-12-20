from vllm import LLM, SamplingParams
import argparse
import json
import pandas as pd

def main(args):
    print("Arguments: ", args)
    print("Loading model...")
    llm = LLM(args.model, tensor_parallel_size=args.tp_size, pipeline_parallel_size=args.pp_size, enable_prefix_caching=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10000)
    print("Model loaded.")

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
    args = argparser.parse_args()
    main(args)

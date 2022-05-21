"""
This code tests using GPT-J locally
"""
import torch
from transformers import AutoTokenizer, GPTJForCausalLM, pipeline

# don't save the transformers in my home directory
import os
os.environ["TRANSFORMERS_CACHE"] = "/data3/benpry/transformers-cache"

if __name__ == "__main__":
    model = GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True,
        cache_dir="/data3/benpry/transformers-cache"
    )
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print(gen("Hi my name is Ben and"))
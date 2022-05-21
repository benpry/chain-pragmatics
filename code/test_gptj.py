"""
This code tests using GPT-J locally
"""
import torch
from transformers import AutoTokenizer, GPTJForCausalLM, pipeline

if __name__ == "__main__":
    model = GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print(gen("Hi my name is Ben and"))
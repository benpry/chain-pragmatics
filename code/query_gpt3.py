"""
This file takes the processed question, asks GPT-3 about it, then saves the response
"""
import os
import time
import pickle
import openai
from pyprojroot import here
from prompt_generation import make_random_k_shot_prompt, make_rationale_prompt

K=5
n_examples = 50
openai.api_key = os.environ["OPENAI_API_KEY"]
task_description = "Choose the most appropriate paraphrase of the first sentence"

if __name__ == "__main__":

    with open(here(f"data/metaphor-corpus/metaphor-corpus-dev.p"), "rb") as fp:
       processed_corpus = pickle.load(fp)

    relevant_corpus = processed_corpus

    for item in relevant_corpus:

        # k_shot_prompt = make_random_k_shot_prompt(item["prompt"], task_description, relevant_corpus, k=K)
        rationale_prompt = make_rationale_prompt(item["prompt"], "differences")

        print(rationale_prompt)

        response = openai.Completion.create(
            engine="text-curie-001",
            prompt=rationale_prompt,
            max_tokens=256,
            n=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        choices = response["choices"]
        print(choices)
        item["model_choices"] = choices

        time.sleep(10)

    with open(here(f"data/model-outputs/gpt3_metaphor_curie_5shot_differences.p"), "wb") as fp:
        pickle.dump(relevant_corpus, fp)

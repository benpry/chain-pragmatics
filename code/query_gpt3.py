"""
This file takes the processed question, asks GPT-3 about it, then saves the response
"""
import os
import pickle
import openai
from pyprojroot import here
from prompt_generation import make_random_k_shot_prompt

K=5
n_examples = 5
openai.api_key = os.environ["OPENAI_API_KEY"]
task_description = "Choose the most appropriate paraphrase of the first sentence"

if __name__ == "__main__":

    with open(here(f"data/metaphor-corpus/processed-metaphor-corpus.p"), "rb") as fp:
       processed_corpus = pickle.load(fp)

    relevant_corpus = processed_corpus[:n_examples]

    for item in relevant_corpus:

        k_shot_prompt = make_random_k_shot_prompt(item["prompt"], task_description, relevant_corpus, k=K)

        print(k_shot_prompt)

        response = openai.Completion.create(
            engine="text-curie-001",
            prompt=k_shot_prompt,
            max_tokens=256,
            n=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )
        choices = response["choices"]

        print(choices)

        item["model_choices"] = choices

    with open(here(f"data/model-outputs/gpt3_metaphor_multiplechoice_curie_5shot.p"), "wb") as fp:
        pickle.dump(relevant_corpus, fp)

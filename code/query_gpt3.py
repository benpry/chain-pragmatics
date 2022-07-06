"""
This file takes the processed question, asks GPT-3 about it, then saves the response
"""
import os
import pandas as pd
import openai
from pyprojroot import here
from prompt_generation import make_k_shot_prompt, make_rationale_prompt

openai.api_key = os.environ["OPENAI_API_KEY"]
task_description = "Choose the most appropriate paraphrase of the first sentence."
prompt_type = "contrast"
gpt_version = "davinci"
corpus_set = "dev"
temp = 0.8
K = 10

gpt_version_codes = {
    "curie": "text-curie-001",
    "davinci": "text-davinci-002"
}

if __name__ == "__main__":

    if corpus_set == "dev":
        df_corpus = pd.read_csv(here(f"data/katz-corpus/katz-corpus-dev.csv"))
    elif corpus_set == "test":
        df_corpus = pd.read_csv(here(f"data/katz-corpus/katz-corpus-test.csv"))
    else:
        raise ValueError(f"Invalid corpus set: {corpus_set}")

    model_choices = []
    for index, row in df_corpus.iterrows():

        if prompt_type == "basic":
            prompt = make_k_shot_prompt(row, task_description, k=K)
        elif prompt_type == "non_explanation":
            prompt = make_rationale_prompt(row["prompt"], task_description, rationale_type=prompt_type,
                                           k=K, step_by_step=False)
        else:
            prompt = make_rationale_prompt(row["prompt"], task_description, rationale_type=prompt_type,
                                           k=K, step_by_step=True)

        response = openai.Completion.create(
             engine=gpt_version_codes[gpt_version],
             prompt=prompt,
             max_tokens=256,
             n=1,
             temperature=temp,
             frequency_penalty=0,
             presence_penalty=0
        )
        choices = response["choices"]
        model_choices.append(choices[0]["text"])

    df_corpus["model_response"] = model_choices

    df_corpus.to_csv(here(f"data/model-outputs/model_responses_set={corpus_set}-gpt={gpt_version}-prompt={prompt_type}-k={K}-temp={temp}.csv"))

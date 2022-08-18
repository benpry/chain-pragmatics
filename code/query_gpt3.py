"""
This file takes the processed question, asks GPT-3 about it, then saves the response
"""
import os
import pandas as pd
import openai
from pyprojroot import here
from prompt_generation import make_k_shot_prompt, make_rationale_prompt

# global variables
openai.api_key = os.environ["OPENAI_API_KEY"]
task_description = "Choose the most appropriate paraphrase of the first sentence."
prompt_type = "options_only"
gpt_version = "curie"
corpus_set = "dev"
temp = 0.7
K = 10

gpt_version_codes = {
    "curie": "text-curie-001",
    "davinci": "text-davinci-002"
}

if __name__ == "__main__":

    # read the relevant corpus
    if corpus_set == "dev":
        df_corpus = pd.read_csv(here(f"data/katz-corpus/katz-corpus-dev.csv"))
    elif corpus_set == "test":
        df_corpus = pd.read_csv(here(f"data/katz-corpus/katz-corpus-test.csv"))
    else:
        raise ValueError(f"Invalid corpus set: {corpus_set}")

    # iterate over examples
    model_choices = []
    for index, row in df_corpus.iterrows():

        # create the relevant prompt
        if prompt_type == "basic":
            prompt = make_k_shot_prompt(row["prompt"], task_description, k=K)
        elif prompt_type == "non_explanation":
            prompt = make_rationale_prompt(row["prompt"], task_description, rationale_type=prompt_type,
                                           k=K, step_by_step=False)
        elif prompt_type == "options_only":
            prompt = make_k_shot_prompt(row["prompt"], task_description, k=K, options_only=True)
        else:
            prompt = make_rationale_prompt(row["prompt"], task_description, rationale_type=prompt_type,
                                           k=K, step_by_step=False)

        print(prompt)

        # get the response from GPT-3
        response = openai.Completion.create(
             engine=gpt_version_codes[gpt_version],
             prompt=prompt,
             max_tokens=256,
             n=1,
             temperature=temp,
             frequency_penalty=0,
             presence_penalty=0
        )
        # add the model's response to model_choices
        choices = response["choices"]
        model_choices.append(choices[0]["text"])

    # save the model choices along with the corpus
    df_corpus["model_response"] = model_choices
    df_corpus.to_csv(here(f"data/model-outputs/model_responses_set={corpus_set}-gpt={gpt_version}-prompt={prompt_type}-k={K}-temp={temp}.csv"))

"""
This file just prints all the different prompts so I can paste them into a word processor to check them for typos
"""
import pandas as pd
from pyprojroot import here
from prompt_generation import make_k_shot_prompt, make_rationale_prompt

# the task description and prompt types to print
task_description = "Choose the most appropriate paraphrase of the first sentence."
prompt_types = ["basic", "non_explanation", "QUD", "similarity", "contrast", "subject_predicate", "options_only"]
K = 10

if __name__ == "__main__":

    df_corpus = pd.read_csv(here(f"data/katz-corpus/katz-corpus-dev.csv"))
    example_test_prompt = df_corpus.sample(1).iloc[0]["prompt"]

    print(example_test_prompt)

    # for each prompt
    for prompt_type in prompt_types:

        # generate the prompt
        if prompt_type == "basic":
            prompt = make_k_shot_prompt(example_test_prompt,
                                        task_description,
                                        k=K
                                        )
        elif prompt_type == "options_only":
            prompt = make_k_shot_prompt(example_test_prompt, task_description, k=K, options_only=True)
        else:
            prompt = make_rationale_prompt(example_test_prompt,
                                           task_description,
                                           rationale_type=prompt_type,
                                           k=K,
                                           step_by_step=False
                                           )


        # print it out
        print("\n" + prompt_type.upper() + "\n")
        print(prompt)

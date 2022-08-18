"""
This file contains code that generates prompts, mainly for few-shot prompting but possibly for other reasons too.
"""
import pandas as pd
import numpy as np
from pyprojroot import here

# constants: the answer markers and rationale indices
answer_markers = ["a)", "b)", "c)", "d)"]
rationale_indices = [1, 5, 6, 9, 11, 13, 17, 21, 22, 25, 28]

# set up the rationale dataframe
df_rationales = pd.read_csv(here("data/prompts/katz/train-rationales.csv"))
df_rationales = df_rationales.iloc[rationale_indices]


def make_rationale_prompt(
        main_question: str,
        task_description: str,
        rationale_type: str = "QUD",
        k: int = 10,
        step_by_step: bool = True
    ) -> str:
    """
    Make a prompt that encourages the model to generate a rationale alongside the answer
    """
    # initialize the prompt with the task description
    prompt = f"{task_description}\n###\n"

    # add each of the k examples
    for index, row in df_rationales.sample(n=k).iterrows():
        prompt += row["prompt"] + "\n"
        if step_by_step:  # in case we want "let's think step by step."
            prompt += "Let's think step by step.\n"

        # add the relevant rationale
        prompt += row[f"{rationale_type}_rationale"] + "\n"
        # denote the end of the exmaple with ###
        prompt += "###\n"

    # add the main question
    prompt += main_question + "\n"
    # add "let's think step by step" if necessary
    if step_by_step:
        prompt += "Let's think step by step.\n"

    return prompt


def make_k_shot_prompt(
        test_prompt: str,
        task_description: str,
        k: int = 10,
        options_only: bool = False
    ) -> str:
    """
    Make a k-shot prompt using the Katz corpus
    """
    # initialize with the task description
    full_prompt = ""
    if not options_only:
        full_prompt += f"{task_description}\n###\n"

    # shuffle the rows and compile the prompts
    df_shots = df_rationales.sample(n=k)
    for index, row in df_shots.iterrows():
        # write the prompt
        if options_only:
            # remove the first line if we are in the options only baseline
            full_prompt += "\n".join(row["prompt"].split("\n")[2:])
        else:
            full_prompt += row["prompt"]
        # write the answer
        full_prompt += f"\nThe answer is {answer_markers[np.argmax(np.fromstring(row['values'][1:-1], dtype=int, sep=' '))]} {row['Good (4)']}"
        full_prompt += "\n###\n"

    # add the test prompt
    if options_only:
        full_prompt += "\n".join(test_prompt.split("\n")[2:])
    else:
        full_prompt += test_prompt

    # add "the answer is"
    full_prompt += "\nThe answer is "

    return full_prompt


def make_katz_prompt(row) -> str:
    """
    This function turns a row of the Katz dataset into a prompt for GPT-3
    """
    # extract the metaphorical statement and all paraphrases
    statement = row["Statement"]
    good = row["Good (4)"]
    less_good = row["Less good (3)"]
    semantic = row["Semantic (2) - category/desription"]
    bad = row["Bad (1)"]

    # shuffle the responses
    responses = [bad, semantic, less_good, good]
    response_indices = np.random.choice(range(len(responses)), size=4, replace=False)

    # create a string with the statement and all the options
    prompt = f'"{statement}"\n\n'
    for marker_idx, i in enumerate(response_indices):
        prompt += answer_markers[marker_idx] + " " + responses[i] + "\n"

    # return the prompt and the goodness scores
    return prompt, response_indices + 1


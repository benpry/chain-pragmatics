"""
This file contains code that generates prompts, mainly for few-shot prompting but possibly for other reasons too.
"""
import pandas as pd
import numpy as np
from preprocess_metaphor_paraphrase_corpus import answer_markers
from pyprojroot import here
from ast import literal_eval

rationale_prompts = {"paraphrase": {}, "katz": {}}
with open(here("data/prompts/paraphrase-corpus/comparison.txt"), "r") as fp:
    rationale_prompts["paraphrase"]["comparison"] = fp.read()
with open(here("data/prompts/paraphrase-corpus/differences.txt"), "r") as fp:
    rationale_prompts["paraphrase"]["differences"] = fp.read()

rationale_indices = [1, 5, 6, 9, 11, 13, 17, 21, 22, 25, 28]
df_katz_rationales = pd.read_csv(here("data/prompts/katz/train-rationales.csv"))
rationale_prompts["katz"] = df_katz_rationales.iloc[rationale_indices]


def make_random_k_shot_prompt(chosen_prompt, task_description, questions, k=5, delimiter="###"):
    """
    Use the questions to make a k-shot prompt with questions and right answers.
    """
    chosen_questions = np.random.choice(questions, size=k, replace=False)

    prompt = task_description + f"\n{delimiter}\n"
    for question in chosen_questions:
        prompt += question["prompt"]
        best_response = answer_markers[np.argmax(question["values"])]
        prompt += f" {best_response}.\n"
        prompt += f"{delimiter}\n"

    prompt += chosen_prompt
    return prompt

def make_rationale_prompt(
        main_question: object,
        task_description: object,
        corpus: object = "katz",
        rationale_type: object = "QUD",
        k: object = 10,
        step_by_step: object = True
    ) -> str:
    """
    Make a prompt that encourages the model to generate a rationale alongside the answer
    """
    prompt = f"{task_description}\n###\n"
    df_qud_rationales = rationale_prompts[corpus]
    for index, row in df_qud_rationales.sample(n=k).iterrows():
        prompt += row["prompt"] + "\n"
        if step_by_step:
            prompt += "Let's think step by step.\n"
        prompt += row[f"{rationale_type}_rationale"] + "\n"
        prompt += "###\n"

    prompt += main_question + "\nLet's think step by step.\n"

    return prompt

def make_katz_prompt(row):
    """
    This function turns a row of the Katz dataset into a prompt for GPT-3
    """
    statement = row["Statement"]
    good = row["Good (4)"]
    less_good = row["Less good (3)"]
    semantic = row["Semantic (2) - category/desription"]
    bad = row["Bad (1)"]
    responses = [bad, semantic, less_good, good]
    response_indices = np.random.choice(range(len(responses)), size=4, replace=False)

    prompt = f'"{statement}"\n\n'
    for marker_idx, i in enumerate(response_indices):
        prompt += answer_markers[marker_idx] + " " + responses[i] + "\n"

    return prompt, response_indices + 1

def make_k_shot_prompt(test_prompt, task_description, k=5):
    """
    Make a k-shot prompt using the Katz corpus
    """
    full_prompt = f"{task_description}\n###\n"
    df_shots = rationale_prompts["katz"].sample(n=k)
    for index, row in df_shots.iterrows():
        # write the prompt
        full_prompt += row["prompt"]
        # write the answer
        full_prompt += f"\nThe answer is {answer_markers[np.argmax(np.fromstring(row['values'][1:-1], dtype=int, sep=' '))]} {row['Good (4)']}"
        full_prompt += "\n###\n"

    full_prompt += test_prompt + "\nThe answer is "

    return full_prompt
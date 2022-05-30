"""
This file contains code that generates prompts, mainly for few-shot prompting but possibly for other reasons too.
"""
import numpy as np
from preprocess_metaphor_corpus import answer_markers
from pyprojroot import here

rationale_prompts = {}
with open(here("data/prompts/comparison.txt"), "r") as fp:
    rationale_prompts["comparison"] = fp.read()
with open(here("data/prompts/differences.txt"), "r") as fp:
    rationale_prompts["differences"] = fp.read()


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

def make_rationale_prompt(question_prompt, rationale_type):
    """
    Make a prompt that encourages the model to generate a rationale alongside the answer
    """
    starter = rationale_prompts[rationale_type]
    return starter + question_prompt

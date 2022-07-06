"""
This file just prints all the different prompts so I can paste them into a word processor to check them for typos
"""

from prompt_generation import make_k_shot_prompt, make_rationale_prompt

task_description = "Choose the most appropriate paraphrase of the first sentence."
prompt_types = ["basic", "non_explanation", "QUD", "similarity", "contrast"]
K = 10

if __name__ == "__main__":

    for prompt_type in prompt_types:
        if prompt_type == "basic":
            prompt = make_k_shot_prompt("",
                                        task_description,
                                        k=K
                                        )
        else:
            prompt = make_rationale_prompt("",
                                           task_description,
                                           rationale_type=prompt_type,
                                           k=K,
                                           step_by_step=False
                                           )

        print("\n" + prompt_type.upper() + "\n")
        print(prompt)

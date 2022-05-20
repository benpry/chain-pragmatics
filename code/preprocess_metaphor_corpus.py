"""
This file reads the metaphor corpus and segments it into prompts and expected answers
"""
import pickle
from pyprojroot import here

def parse_question(question):
    lines = question.strip().split("\n")
    metaphor = lines[0].strip()
    answers = []
    values = []
    for answer in lines[1:]:
        x, y = answer.split("#")
        answers.append(x.strip())
        values.append(y.strip())

    return {
        "metaphor": metaphor,
        "paraphrases": answers,
        "values": values
    }

answer_markers = ["a)", "b)", "c)", "d)"]

def write_prompt(parsed_question):

    prompt = f'"{parsed_question["metaphor"]}"\nWhich of these is an accurate paraphrase of the above sentence?'
    for i, paraphrase in enumerate(parsed_question["paraphrases"]):
        prompt += f"\n{answer_markers[i]} {paraphrase}"
    prompt += "\n\nThe answer is"

    return prompt

if __name__ == "__main__":

    # read the corpus file
    with open(here("data/metaphor-corpus/metaphor_paraphrase_corpus.txt"), "r") as fp:
        corpus = fp.read()

    questions = corpus.strip().split("\n\n")

    parsed_questions = []
    for question in questions:
        try:
            parsed = parse_question(question)
            if len(parsed["values"]) > 0:
                parsed_questions.append(parsed)
        except ValueError:
            print(f"failed to parse {question}")

    # compile prompt
    for question in parsed_questions:
        question["prompt"] = write_prompt(question)

    with open(here("data/metaphor-corpus/processed-metaphor-corpus.p"), "wb") as fp:
        pickle.dump(parsed_questions, fp)

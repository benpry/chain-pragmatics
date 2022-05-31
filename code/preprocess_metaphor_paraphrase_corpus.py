"""
This file reads the metaphor corpus and segments it into prompts and expected answers
"""
from random import shuffle
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

train_dev_test = (0.7, 0.15, 0.15)

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


    # make train-test split
    shuffle(parsed_questions)

    train_prop, dev_prop, test_prop = train_dev_test
    sets = {}
    sets["train"] = parsed_questions[:int(train_prop * len(parsed_questions))]
    sets["dev"] = parsed_questions[len(sets["train"]):len(sets["train"]) + int(dev_prop * len(parsed_questions))]
    sets["test"] = parsed_questions[len(sets["train"]) + len(sets["dev"]):]

    for name, set in sets.items():
        print(f"{name} set length: {len(set)}")
        with open(here(f"data/metaphor-corpus/metaphor-corpus-{name}.p"), "wb") as fp:
            pickle.dump(set, fp)

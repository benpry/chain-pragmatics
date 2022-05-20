"""
This file takes the model outputs and analyzes them
"""
import pickle
import numpy as np
from pyprojroot import here

guess_labels = ["a", "b", "c", "d"]
def extract_guess(response):
    """
    Figure out which response the model chose
    """
    response = response.strip().lower()
    if response[0] in guess_labels:
        return guess_labels.index(response[0])
    else:
        return None


def rank_guess(guess, ratings):
    """
    Given a guess and list of ratings, get the rating corresponding to the chosen guess
    """
    return int(ratings[guess])


def random_baseline(questions):
    """
    Suppose we randomly selected answers, what ranks would we end up with?
    """
    rand_ratings = []
    for question in questions:
        random_rating = int(np.random.choice(question["values"]))
        rand_ratings.append(random_rating)

    return rand_ratings

if __name__ == "__main__":

    with open(here(f"data/model-outputs/gpt3_metaphor_multiplechoice_curie_5shot.p"), "rb") as fp:
        question_responses = pickle.load(fp)

    non_parsed_guesses = 0
    guess_ranks = []
    for question in question_responses:

        guess = extract_guess(question["model_choices"][0]["text"])
        if guess is None:
            non_parsed_guesses += 1
            continue

        rank = rank_guess(guess, question["values"])
        guess_ranks.append(rank)
    mean_guess_rank = np.mean(guess_ranks)

    print(f"{non_parsed_guesses} guesses not parsed")
    print(f"mean rank: {mean_guess_rank}")

    random_means = []
    for i in range(10000):
        random_ratings = random_baseline(question_responses)
        random_means.append(np.mean(random_ratings))
    p_val = len([m for m in random_means if m > mean_guess_rank]) / len(random_means)
    print(f"mean random rating {np.mean(random_means)}")
    print(f"p-value: {p_val}")

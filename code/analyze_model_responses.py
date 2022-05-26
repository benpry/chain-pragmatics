"""
This file takes the model outputs and analyzes them
"""
import pickle
import numpy as np
from pyprojroot import here
import seaborn as sns
import matplotlib.pyplot as plt

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


def bootstrapped_ci(scores, n=100000):
    all_bs = np.random.choice(scores, size=(n, len(scores)))
    means = np.mean(all_bs, axis=1)
    mean = np.mean(means)
    ci_lower = np.percentile(means, 2.5)
    ci_upper = np.percentile(means, 97.5)

    return mean, ci_lower, ci_upper

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

    with open(here(f"data/model-outputs/gpt3_metaphor_multiplechoice_curie_5shot_comparison.p"), "rb") as fp:
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

    mean, ci_lower, ci_upper = bootstrapped_ci(guess_ranks)
    print(f"mean rank: {mean}, [{ci_lower}, {ci_upper}]")

    print(f"{non_parsed_guesses} guesses not parsed")

    random_means = []
    for i in range(10000):
        random_ratings = random_baseline(question_responses)
        random_means.append(np.mean(random_ratings))
    random_ci_lower = np.percentile(random_means, 2.5)
    random_ci_upper = np.percentile(random_means, 97.5)
    p_val = len([m for m in random_means if m > mean]) / len(random_means)
    print(f"mean random rating {np.mean(random_means)}, [{random_ci_lower}, {random_ci_upper}]")
    print(f"p-value: {p_val}")

    print(guess_ranks)
    hist = sns.histplot(guess_ranks, discrete=True)
    hist.set_title("Response Appropriateness Distribution: Curie with 5-shot comparison prompts")
    hist.set_xticks([1, 2, 3, 4])
    hist.set_xlabel("Appropriateness Score")
    hist.get_figure().savefig(here("figures/appropriateness_distribution_curie_comparison.png"))

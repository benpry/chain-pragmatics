"""
This file takes the model outputs and analyzes them
"""
import pickle
import re

import numpy as np
from pyprojroot import here
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

guess_labels = ["a", "b", "c", "d"]
def extract_guess(response):
    """
    Figure out which response the model chose
    """
    response = response.strip().lower()
    match = re.findall(r"the answer is ([a-d])", response)
    if len(match) == 0:
        return None
    else:
        return guess_labels.index(match[0])


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

def random_baseline(ratings, n_questions=100):
    """
    Suppose we randomly selected answers, what ranks would we end up with?
    """
    rand_ratings = []
    for q in range(n_questions):
        random_rating = int(np.random.choice(ratings))
        rand_ratings.append(random_rating)

    return rand_ratings

corpus = "katz"
gpt_version = "curie"
prompt_type = "QUD"
K = 10

if __name__ == "__main__":

    df_responses = pd.read_csv(here(f"data/model-outputs/model_responses_corpus={corpus}-gpt={gpt_version}-prompt={prompt_type}-k={K}.csv"))
    non_parsed_guesses = 0
    guess_ranks = []
    for index, row in df_responses.iterrows():

        if not isinstance(row["model_response"], str):
            non_parsed_guesses += 1
            print("nan response")
            continue

        guess = extract_guess(row["model_response"])
        if guess is None:
            non_parsed_guesses += 1
            print(f"couldn't parse guess: {row['model_response']}")
            continue

        rank = rank_guess(guess, np.fromstring(row["values"][1:-1], sep=" "))
        guess_ranks.append(rank)

    mean, ci_lower, ci_upper = bootstrapped_ci(guess_ranks)
    print(f"mean rank: {mean}, [{ci_lower}, {ci_upper}]")

    print(f"{non_parsed_guesses} guesses not parsed")

    random_means = []
    for i in range(10000):
        random_ratings = random_baseline([1, 2, 3, 4])
        random_means.append(np.mean(random_ratings))
    random_ci_lower = np.percentile(random_means, 2.5)
    random_ci_upper = np.percentile(random_means, 97.5)
    p_val = len([m for m in random_means if m > mean]) / len(random_means)
    print(f"mean random rating {np.mean(random_means)}, [{random_ci_lower}, {random_ci_upper}]")
    print(f"p-value: {p_val}")

    print(guess_ranks)
    hist = sns.histplot(guess_ranks, discrete=True)
    hist.set_title(f"Response Appropriateness Distribution: {corpus} corpus {gpt_version} with {K}-shot {prompt_type} prompts",
                   fontsize=10)
    hist.set_xticks([1, 2, 3, 4])
    hist.set_xlabel("Appropriateness Score")
    hist.get_figure().savefig(here(f"figures/appropriateness_distribution_corpus={corpus}-gpt={gpt_version}-prompt={prompt_type}-k={K}.png"))

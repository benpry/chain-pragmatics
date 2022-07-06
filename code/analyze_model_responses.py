"""
This file takes the model outputs and analyzes them
"""
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
        match = re.findall(r"the speaker is saying ([a-d])\)", response)
        if len(match) == 0:
            if len(response) < 10:
                match = re.findall(r"[a-d]", response)
                if len(match) == 0:
                    return None
            else:
                return None
    return match[0]


def rank_guess(guess, ratings):
    """
    Given a guess and list of ratings, get the rating corresponding to the chosen guess
    """
    return int(ratings[guess_labels.index(guess)])


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

corpus_set = "test"
gpt_version = "curie"
prompt_type = "basic"
K = 10
temp = 0.9

prompt_title_names = {
    "QUD_v3": "QUD"
}

if __name__ == "__main__":

    df_responses = pd.read_csv(here(f"data/model-outputs/model_responses_set={corpus_set}-gpt={gpt_version}-prompt={prompt_type}-k={K}-temp={temp}.csv"))
    non_parsed_guesses = 0
    guess_ranks = []
    raw_guesses = []

    print(f"Analyzing {len(df_responses)} responses")

    for index, row in df_responses.iterrows():

        if not isinstance(row["model_response"], str):
            non_parsed_guesses += 1
            guess_ranks.append(None)
            raw_guesses.append(None)
            print("nan response")
            continue

        guess = extract_guess(row["model_response"])
        if guess is None:
            non_parsed_guesses += 1
            guess_ranks.append(None)
            raw_guesses.append(None)
            print(f"couldn't parse guess: {row['model_response']}")
            continue

        rank = rank_guess(guess, np.fromstring(row["values"][1:-1], sep=" "))
        guess_ranks.append(rank)
        raw_guesses.append(guess)

    df_responses["appropriateness_score"] = guess_ranks
    df_responses["raw_guess"] = raw_guesses
    df_responses.to_csv(here(f"data/model-outputs/model_responses_set={corpus_set}-gpt={gpt_version}-prompt={prompt_type}-k={K}-temp={temp}-processed.csv"))

    guess_ranks = [x for x in guess_ranks if x is not None]
    raw_guesses = [x for x in raw_guesses if x is not None]

    mean, ci_lower, ci_upper = bootstrapped_ci(guess_ranks)
    print(f"mean rank: {mean}, [{ci_lower}, {ci_upper}]")

    print(f"{non_parsed_guesses} guesses not parsed")

    random_means = []
    for i in range(10000):
        random_ratings = random_baseline([1, 2, 3, 4], n_questions=len(guess_ranks))
        random_means.append(np.mean(random_ratings))
    random_ci_lower = np.percentile(random_means, 2.5)
    random_ci_upper = np.percentile(random_means, 97.5)
    p_val = len([m for m in random_means if m > mean]) / len(random_means)
    print(f"mean random rating {np.mean(random_means)}, [{random_ci_lower}, {random_ci_upper}]")
    print(f"p-value: {p_val}")

    if prompt_type in prompt_title_names:
        prompt_title = prompt_title_names[prompt_type]
    else:
        prompt_title = prompt_type

    print(guess_ranks)
    hist = sns.histplot(guess_ranks, discrete=True)
    hist.set_title(f"Response Appropriateness Distribution: {gpt_version} with {K}-shot {prompt_title} prompts",
                   fontsize=10)
    hist.set_xticks([1, 2, 3, 4])
    hist.set_xlabel("Appropriateness Score")
    hist.get_figure().savefig(here(f"figures/appropriateness_distribution_gpt={gpt_version}-prompt={prompt_type}-k={K}-temp={temp}.png"))

    plt.clf()

    print(raw_guesses)
    hist = sns.countplot(sorted(raw_guesses))
    hist.set_title(f"Response Distribution: {gpt_version} with {K}-shot {prompt_title} prompts",
                   fontsize=10)
    hist.set_xlabel("Response")
    hist.get_figure().savefig(here(f"figures/response_distribution_gpt={gpt_version}-prompt={prompt_type}-k={K}-temp={temp}.png"))

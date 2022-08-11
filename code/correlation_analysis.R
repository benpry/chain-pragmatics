library(tidyverse)
library(brms)
library(here)
library(broom)
library(Hmisc)

# constants
corpus = "katz"
corpus_set = "test"
gpt_version = "davinci"
prompt_types = c("basic", "non_explanation", "QUD_v3", "similarity", "contrast")
K = 10
temp = 0.9

df.all_responses <- data.frame()
for (prompt_type in prompt_types) {
  df.responses_prompt <- read.csv(
    here(sprintf("data/model-outputs/model_responses_corpus=%s-set=%s-gpt=%s-prompt=%s-k=%s-temp=%s-processed.csv",
                 corpus, corpus_set, gpt_version, prompt_type, K, temp)))
  
  df.responses_prompt <- df.responses_prompt |>
    mutate(prompt_type = prompt_type)
  
  df.all_responses <- rbind(df.all_responses, df.responses_prompt)
}

df.all_responses <- df.all_responses |>
  select(ID, Group, Statement, prompt_type, appropriateness_score)

df.responses_wide <- df.all_responses |>
  pivot_wider(
    id_cols = c("ID", "Group", "Statement"),
    names_from = prompt_type,
    values_from = appropriateness_score
  )

cor.test(df.responses_wide$basic, df.responses_wide$non_explanation)

corr_results <- df.responses_wide |>
  select(basic, non_explanation, QUD_v3, similarity, contrast) |>
  as.matrix() |>
  rcorr(type = "pearson")

rs = corr_results[["r"]]
ns = corr_results[["n"]]
ps = corr_results[["P"]]

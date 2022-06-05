library(tidyverse)
library(brms)
library(here)
library(broom)
library(Hmisc)

# constants
corpus = "katz"
gpt_version = "curie"
prompt_type = "QUD_v3"
K = 10
temp = 0.9

df.responses_rationale <- read.csv(
  here(sprintf("data/model-outputs/model_responses_corpus=%s-gpt=%s-prompt=%s-k=%s-temp=%s-processed.csv",
               corpus, gpt_version, prompt_type, K, temp)))

df.responses_basic <- read.csv(
  here(sprintf("data/model-outputs/model_responses_corpus=%s-gpt=%s-prompt=%s-k=%s-temp=%s-processed.csv",
               corpus, gpt_version, "basic", K, temp)))

df.differences <- df.responses_rationale |>
  select(ID, appropriateness_score) |>
  rename(appropriateness_rationale = appropriateness_score) |>
  left_join(df.responses_basic, on=ID) |>
  rename(appropriateness_basic = appropriateness_score) |>
  mutate(appropriateness_diff = appropriateness_rationale - appropriateness_basic) |>
  mutate(appropriateness_rationale_norm = appropriateness_rationale - 2.5)

df.concatenated <- df.responses_rationale |>
  select(ID, appropriateness_score) |>
  mutate(prompt_type = "rationßale") |>
  rbind(df.responses_basic |> 
          select(ID, appropriateness_score) |>
          mutate(prompt_type = "basic")
        )

model <- brm(appropriateness_diff ~ 1, data=df.differences)
summary(model)

psycholing_cols <- c("CMP","ESI","MET","MGD","IMG","IMS","IMP","FAM","SRL","ALT")
model <- brm(appropriateness_rationale ~ ALT, family=cumulative(), data=df.differences)
summary(model)

interesting_looking_measures <- c("ESI", "IMG", "IMP", "FAM")
model <- brm(appropriateness_rationale ~ FAM, family=cumulative(), data=df.differences)
summary(model)
plot(model)

df.differences |>
  select(interesting_looking_measures) |>
  as.matrix() |>
  rcorr(type="pearson")

model <- brm(appropriateness_score ~ 1 + prompt_type, family=cumulative(), data = df.concatenated)
summary(model)

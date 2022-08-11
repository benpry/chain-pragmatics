library(tidyverse)
library(brms)
library(here)
library(broom)

# constants
corpus = "katz"
corpus_set = "test"
gpt_version = "curie"
prompt_type = "contrast"
K = 10
temp = 0.9

df.responses_rationale <- read.csv(
  here(sprintf("data/model-outputs/model_responses_corpus=%s-set=%s-gpt=%s-prompt=%s-k=%s-temp=%s-processed.csv",
               corpus, corpus_set, gpt_version, prompt_type, K, temp)))

df.responses_basic <- read.csv(
  here(sprintf("data/model-outputs/model_responses_corpus=%s-set=%s-gpt=%s-prompt=%s-k=%s-temp=%s-processed.csv",
               corpus, corpus_set, gpt_version, "basic", K, temp)))

df.differences <- df.responses_rationale |>
  select(ID, appropriateness_score) |>
  rename(appropriateness_rationale = appropriateness_score) |>
  left_join(df.responses_basic, on=ID) |>
  rename(appropriateness_basic = appropriateness_score) |>
  mutate(appropriateness_diff = appropriateness_rationale - appropriateness_basic) |>
  mutate(appropriateness_rationale_norm = appropriateness_rationale - 2.5)

df.concatenated <- df.responses_rationale |>
  select(ID, appropriateness_score) |>
  mutate(prompt_type = "rationale") |>
  rbind(df.responses_basic |> 
          select(ID, appropriateness_score) |>
          mutate(prompt_type = "basic")
        )

top_30_familiar <- slice_max(df.differences, order_by=FAM, n=30)
bottom_30_familiar <- slice_min(df.differences, order_by=FAM, n=30)

mean_fam_difference = mean(top_30_familiar$appropriateness_rationale, na.rm=T) - mean(bottom_30_familiar$appropriateness_rationale, na.rm=T)

psycholing_cols <- c("CMP","ESI","MET","MGD","IMG","IMS","IMP","FAM","SRL","ALT")
model <- brm(appropriateness_rationale ~ FAM, family=cumulative(), data=df.differences)
summary(model)

interesting_looking_measures <- c("ESI", "IMG", "IMP", "FAM")
model <- brm(appropriateness_basic ~ FAM, family=cumulative(), data=df.differences)
summary(model)
plot(model)

model <- brm(appropriateness_score ~ prompt_type, family=cumulative(), data = df.concatenated)
summary(model)

ggplot(
  data = df.differences,
  mapping = aes(x = appropriateness_basic)
) +
  geom_histogram()
  
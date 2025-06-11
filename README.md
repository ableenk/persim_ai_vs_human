# AI vs Human text classification with persistent image

## requirements

numpy, pandas, sklearn, gensim, xgboost, ripser, persim



## dataset

3000 texts created by human and 3000 texts created by AI

generated column: 0 is for human and 1 is for AI

data is already processed, all texts are lowercase and have no punctuation

# method
for each text:
1. take 120 most frequent words
2. vectorize all of them
3. calculate persistent diagram
4. calculate persistence image of this diagram
5. train models with persistence image as feature

# result
score on test: 0.85

only 4500 texts used for training



# reference
original dataset: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text/data
persim paper: https://arxiv.org/abs/1507.06217

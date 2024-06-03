# SMM4H2024-Task3-CODES

This repository contains resources used as part of participation in the ACL SMM4H: The 9th Social Media Mining for Health Research and Applications Workshop and Shared Tasks. The specific focus of this repository is Task 3, which focuses on the generalizability of large language models for social media PLN, in particular on the analysis of social media user-generated content to study health-related outcomes in the context of social anxiety.

## Introduction

Social anxiety disorder (SAD), an anxiety disorder whose onset appears mostly in early adolescence and may affect up to 12% of the population at some point of their lives. About one-third of people with SAD report experiencing symptoms for 10 years before seeking treatment, however, people do turn to social media outlets, such as Reddit, to discuss their symptoms and share or ask other users about what may help alleviate these symptoms. While, as has been found with other anxiety disorders, being outdoors in green or blue spaces may be beneficial for relieving symptoms, scant research exists into the effect of these on SAD. In order to qualitatively assess the effects of outdoor spaces, posts that mention these locations and the user's sentiment towards them must be identified for further study.

This task presents a multi-class classification task to categorize posts that mention one or more pre-determined keywords related to outdoor spaces into one of four categories: 1 = positive effect, 2 = neutral or no effect, 3 = negative effect, or 4 = unrelated, where the keyword mention is not referencing an actual outdoor space or activity, or their own interaction with the outdoor space. Details for each class can be found in the associated annotation guidelines (TBA). There is only one category per post. This task has 3,000 annotated posts which were downloaded from the r/socialanxiety subreddit and filtered first to only include users between the ages of 12 and 25, and then for the mention of one of 80 keywords related to green or blue spaces. 80% of the data will be made available for training and validation, and 20% of the data will be held out for evaluation. The evaluation metric for this task is the macro-averaged F1-score over all 4 classes. The data include annotated collections of Reddit posts which will be shared in csv files. There are 4 fields in the csv files: post_id, keyword, text, label. 

## Dataset
The data provided are divided as follows:

- Training set: 1800 annotated posts.
- Validation set: 600 annotated posts.
- Test set: 1200 unannotated posts.
- Evaluation metric: macro-averaged F1-score

## Structure

The repository is organized as follows:

- `DATA_PROVIDED`: This directory contains the data used for the task.
- `DATA_EXPLORATION`: This directory contains the codes for the exploratory study of the training and validation data provided. A .pdf file is also included where a summary of the results obtained in the analysis is included.
- `NAIVE_BAYES`: This directory contains the codes for the training and validation of naive Bayes models of Bernoulli and multinomial type where they are differentiated by the PLN techniques used in them, in some cases using lemmatized texts or stemming, using PLN libraries such as Spacy or NLTK. These models are trained to classify in 4 classes and also in binary classification for the proposed cascade classification. A .pdf file is also included where a summary of the results obtained in the different models is included.
- `BEST_CLASSIFICATION_NAIVE+SENTIMENT_ANALYSIS_MODELS`: This directory contains the code of the best result obtained in the cascade classification using, for the first classification (binary), a naive bayes model, while the second classification is done with a raw HuggingFace transformer model (without fine tuning).
- `FIRST_CLASSIFICATION_MODEL_SEARCH`: This directory contains the codes used to search for the best zero shot learning model for the first stage of the cascade classification (binary classification), and also includes the search for the best model for the classification in the 4 classes.
- `FIRST_CLASSIFICATION_BEST_MODEL_FINDED_FINE_TUNING`: This directory contains the codes by which the best zero shot learning model found were fine tuned for the first stage of the cascade classification (binary classification).
- `SECOND_CLASSIFICATION_SENTIMENT_ANALYSIS_MODEL_SEARCH`: This directory contains the codes used to find the best sentiment analysis model for the second stage of the cascade classification (classification into 3 classes positive, negative, neutral).
- `SECOND_CLASSIFICATION_SENTIMENT_ANALYSIS_MODELS_FINE_TUNING`: This directory contains the codes by which the 2 best sentiment analysis models found were fine tuned for the second stage of the cascade classification (classification into 3 classes positive, negative, neutral).
- `BEST_CLASSIFICATION_ZLS+SENTIMENT_ANALYSIS_MODELS`: This directory contains 2 codes of the bests results obtained in the cascade classification using, for the first classification (binary), a zero shot learning fine tuned model, while the second classification is done with a sentiment analysis fine tuned models.
- `SECOND_CLASSIFICATION_ZSL_MODEL_SEARCH`: This directory contains the codes used to search for the best zero shot learning model for the second stage of the cascade classification (classification into 3 classes positive, negative, neutral).
- `SECOND_CLASSIFICATION_BEST_ZSL_MODELS_FINE_TUNING`: This directory contains the codes by which the 2 bests zero shot learning model found were fine tuned for the second stage of the cascade classification (classification into 3 classes positive, negative, neutral).
- `FINAL_PREDICTIONS_SENDED`: This directory contains the 3 .csv final files that were presented to the competition.


## Requirements

To run the code in this repository, you'll need the following dependencies:

- Python 3
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- NLTK
- Spacy


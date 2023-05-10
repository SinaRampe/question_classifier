# Question Classifier

Question Classifier is a machine learning model that classifies whether a given question is related to health topics or not. This repository contains the code and data to train and test the classifier.

For the understanding of the problem and the creation of the code, the book "Natural Language Processing with Transformers" (O'Reilly) was consulted. The corresponding Jupyter notebook can be accessed here: https://github.com/nlp-with-transformers/notebooks/blob/main/02_classification.ipynb

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)

## Overview

The purpose of this project is to create a classifier that can accurately identify questions related to health topics. This can be useful for various applications, such as filtering non-healthcare-related queries from a question-answering system, assisting in medical chatbots, or categorizing webcraped content.

The model is built using Python and the DistilBERT base model (uncased) witch is a distilled version of the BERT base model.

## Dataset

The dataset used in this project is a self composed, labeled and balanced dataset. I used 1900 hand-labeled questions from the Stanford Question Answering Dataset (of which 80 were health questions) and 1740 questions from other healthcare-Q&A-Datasets like AskDocs https://github.com/ju-resplande/askD and others, that can be found here https://github.com/LasseRegin/medical-question-answer-data.

The dataset is divided into training and testing sets, which can be found in the `data` folder of this repository.




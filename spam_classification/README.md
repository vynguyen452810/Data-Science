# Spam Email Classification

## Problem
Spam emails, also known as junk mail, are a common nuisance, often containing cryptic messages, scams, or phishing attempts. The challenge lies in accurately distinguishing between spam and legitimate ("ham") emails based on their content.

## Approach
My approach involves implementing a pipeline to convert each email into a feature vector, enabling machine learning models to classify them effectively. This pipeline includes steps such as data exploration, preprocessing, and model training.

## Data Exploratory and Preprocessing
1. Explore email content:
    * Investigate the structure and content of the emails.
2. Data Preprocessing:
    * Utilize BeautifulSoup to extract text from HTML-formatted emails.
    * Leverage the Natural Language Toolkit (NLTK) for word stemming, reducing words to their base forms.
    * Replace URLs with the word "URL" to standardize them and remove identifying information.

## Building the Pipeline to Convert Emails into Feature Vectors

* Email to Word Counter Transformer:
    * Tokenize the text (split it into words).
    * Perform preprocessing steps such as lowercasing, replacing URLs and numbers, removing punctuation, and stemming(reduce words to their base form).
    * Output a numpy array where each element is a dictionary representing word counts in each email.
* Word Counter to Vector Transformer:
    *  Calculate the most common words and assign each word an index.
    *  Limit the count of each word to a maximum of 10 occurrences to prevent dominance of less informative words.
    *  Convert word counts into a sparse matrix representation, where each row represents an email and each column represents a word in the vocabulary.

## Training and Results
I trained several models on the processed data and evaluated their performance:
* Logistic Regression: Accuracy=0.9846, Precsion=0.9394, Recall=0.9789
* Support Vector Machine: Accuracy=0.9763, Precsion=0.8990, Recall=0.9368
* MLPClassifier: Accuracy=0.9854, Precsion=0.9787, Recall=0.9684

# Usage

## Installation
To use this project, ensure you have the following libraries installed:
* Pandas
* Numpy
* Scikit-learn (sklearn)
* urllib
* Scipy
* BeautifulSoup
* nltk

You can install these libraries using pip, Python's package manager. Here's the command to install them:
    !pip install pandas numpy scikit-learn urllib scipy BeautifulSoup nltk

## Preparing the datasets
* Load data from the source:
http://spamassassin.apache.org/old/publiccorpus/

## Running the Code
* Run the provided notebook (spam_classification.ipynb) to download, preprocess data, train models, and obtain results. Ensure all dependencies are installed beforehand.

import numpy as np
from collections import Counter
import re

def load_data():
    """Load data from file."""
    with open('MultinomialNaiveBayes/messages.txt') as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    return data


def make_Dictionary(train_data):
    """Create a dictionary of words from the training set."""
    all_words = []
    for mail in train_data:
        words = mail.split()
        all_words += words
    dictionary = Counter(all_words)
    # list_to_remove = dictionary.keys()
    # for item in list_to_remove: # this works with python 2.x version
    for item in list(dictionary): # this works with python 3.x version
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary


def extract_features(data, dictionary):
    """Extract features from the data."""
    features_matrix = np.zeros((len(data), 3000))
    docID = 0
    for mail in data:
        words = mail.split()
        for word in words:
            wordID = 0
            for i, d in enumerate(dictionary):
                if d[0] == word:
                    wordID = i
                    features_matrix[docID, wordID] = words.count(word)
        docID = docID + 1
    return features_matrix


def fit(train_matrix, train_spam, train_ham, dictionary):
    """
    Fit the model to the training data.
    
    Parameters
    ----------
    train_matrix : numpy array, Training data.
    train_spam : list, List of spam emails.
    train_ham : list, List of ham emails.
    dictionary : list, List of words in the dictionary.
            
    Returns
    -------
    phi_spam : numpy array, Probability of each word in the dictionary given spam.
    phi_ham : numpy array, Probability of each word in the dictionary given ham.
    phi_Y : float, Probability of spam.
    """
    n_words = len(dictionary)

    phi_spam = np.zeros(n_words)
    phi_ham = np.zeros(n_words)

    spam_word_count = np.sum(train_matrix[:len(train_spam)], axis=0)
    ham_word_count = np.sum(train_matrix[len(train_spam):], axis=0)

    print(train_matrix[:len(train_spam)])
    print(train_matrix[len(train_spam):])

    phi_spam = (spam_word_count + 1) / (np.sum(spam_word_count) + n_words)
    phi_ham = (ham_word_count + 1) / (np.sum(ham_word_count) + n_words)
    print(phi_spam)
    print(phi_ham)

    phi_Y = len(train_spam) / len(train_matrix)

    return phi_spam, phi_ham, phi_Y


def predict(test_matrix, theta_spam, theta_ham, phi_Y):
    """
    Predict the class of each email in the test set.

    Parameters
    ----------
    test_matrix : numpy array, Test data.
    theta_spam : numpy array, Probability of each word in the dictionary given spam.
    theta_ham : numpy array, Probability of each word in the dictionary given ham.
    phi_Y : float, Probability of spam.

    Returns
    -------
    y_pred : numpy array, Predicted class of each email in the test set.
    """

    y_pred = np.zeros(len(test_matrix))
    log_prior_spam = np.log(phi_Y)
    log_prior_ham = np.log(1 - phi_Y)

    for i in range(len(test_matrix)):
        log_likelihood_spam = np.sum(np.log(theta_spam) * test_matrix[i])
        log_likelihood_ham = np.sum(np.log(theta_ham) * test_matrix[i])

        log_posterior_spam = log_likelihood_spam + log_prior_spam
        log_posterior_ham = log_likelihood_ham + log_prior_ham

        if log_posterior_spam > log_posterior_ham:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred


data = load_data()

split = int(len(data)*0.8)

train_data = data[:split]
test_data = data[split:]

train_spam = []
train_ham = []
for i in range(len(train_data)):
    if train_data[i].split('\t')[0] == 'spam':
        train_spam.append(train_data[i].split(' ')[1])
    else:
        train_ham.append(train_data[i].split('\t')[1])


test_spam = []
test_ham = []
for i in range(len(test_data)):
    if test_data[i].split('\t')[0] == 'spam':
        test_spam.append(test_data[i].split('\t')[1])
    else:
        test_ham.append(test_data[i].split('\t')[1])

dictionary = make_Dictionary(train_spam + train_ham)

train_matrix = extract_features(train_spam + train_ham, dictionary)
test_matrix = extract_features(test_spam + test_ham, dictionary)


phi_spam, phi_ham, phi_y = fit(train_matrix, train_spam, train_ham, dictionary)

y_pred = predict(test_matrix, phi_spam, phi_ham, phi_y)


TP = 0
FP = 0
TN = 0
FN = 0

for i in range(len(y_pred)):
    if y_pred[i] == 1 and test_data[i].split('\t')[0] == 'spam':
        TP += 1
    elif y_pred[i] == 1 and test_data[i].split('\t')[0] == 'ham':
        FP += 1
    elif y_pred[i] == 0 and test_data[i].split('\t')[0] == 'ham':
        TN += 1
    elif y_pred[i] == 0 and test_data[i].split('\t')[0] == 'spam':
        FN += 1

confusion_matrixs = np.array([[TP, FP], [FN, TN]])
print("confusion matrix:",  confusion_matrixs)

accuracy = (TP + TN) / (TP + FP + TN + FN)
print("accuracy:", accuracy)

precision = TP / (TP + FP)
print("precision:", precision)

recall = TP / (TP + FN)
print("recall:", recall)

F1 = 2 * precision * recall / (precision + recall)
print("F1 score:", F1)

import numpy as np
import pandas as pd
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import pandas, xgboost, numpy, textblob, string
from textblob import TextBlob


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return predictions


def predict_labels(herry_data):
    model_data = pd.read_excel('rest_data_delimited.xlsx', encoding='utf-8', sheet_name = 'Sheet1')
    model_data = model_data.dropna(axis=0)
    raw_X = np.array(model_data['answer_sentence'])
    raw_y = list(model_data['sentence_epa_spss'])
    y = []
    EPA_list = ['<EPA1>', '<EPA2>', '<EPA3>', '<EPA4>', '<EPA5>', '<EPA6>', '<EPA7>', '<EPA9>', '<EPA12>', '<EPA13>', '<Trustworthiness>']
    for i in raw_y:
        temp_y = [0 for i in range(11)]
        for j in range(11):
            if EPA_list[j] in i:
                temp_y[j] = 1
        y.append(temp_y)
    y = np.array(y)
    # load the dataset
    data = open('corpus').read()
    labels, texts = [], []
    for i, line in enumerate(data.split("\n")):
        content = line.split()
        labels.append(content[0])
        texts.append(" ".join(content[1:]))
    # create a dataframe using texts and lables
    trainDF = pandas.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels
    # train_x, valid_x, train_y, valid_y = model_selection.train_test_split(raw_X, y)
    split = int(len(raw_X)*0.9) + 9
    train_x = raw_X[:split]
    train_y = y[:split]
    valid_x = raw_X[split:]
    valid_y = y[split:]

    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(trainDF['text'])

    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)
    # herry_data = pd.read_excel('harry_potter_sentence_with_raw.xlsx', encoding='utf-8')
    test_x = herry_data["answer_sentence"]
    xtest_count = count_vect.transform(test_x)

    # DT on Count Vectors
    pred = train_model(OneVsRestClassifier(DecisionTreeClassifier()), xtrain_count, train_y, xtest_count)
    comment_ids = []
    answer_sentences = []
    sentence_locations = []
    epa_pred = []
    sentiment_pred = []
    for i, row in herry_data.iterrows():
        # if i >= 100:
        #     continue
        if sum(pred[i,:]) == 0:
            comment_ids.append(row['comment_id'])
            answer_sentences.append(row['sentence_raw'])
            sentence_locations.append(row['sentence_location'])
            epa_pred.append("No EPA Found")
            sentiment_pred.append((TextBlob(row['answer_sentence']).sentiment[1] - 0.5) * 2)
            continue
        for j in range(11):
            if pred[i][j] == 1:
                comment_ids.append(row['comment_id'])
                answer_sentences.append(row['sentence_raw'])
                sentence_locations.append(row['sentence_location'])
                epa_pred.append(EPA_list[j])
                sentiment_pred.append((TextBlob(row['answer_sentence']).sentiment[1] - 0.5) * 2)
    ans_data = {'comment_id' : comment_ids, 'answer_sentence' : answer_sentences, 'sentence_location' : sentence_locations, 'EPA_pred' : epa_pred, 'Sentiment_pred' : sentiment_pred}
    df_ans = pd.DataFrame(ans_data)
    df_ans = df_ans[['comment_id', 'answer_sentence', 'sentence_location', 'EPA_pred', 'Sentiment_pred']]
    return df_ans



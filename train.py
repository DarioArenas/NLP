import os
import sys
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from utils import read_txt, preprocess_data


def train_classifier(path):

    # set paths
    os.chdir(path)
    categories = os.listdir()

    # read data
    data = load_data(categories)
    data = data.reset_index(drop=True)
    data.index.name = 'id'

    # preprocess data
    data = preprocess_data(data)

    # set label
    data = set_label(data, categories)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        data['text_4'],
        data['category'],
        test_size=0.15,
        random_state=8
    )

    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words='english')
    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(X_train)

    # Train Multinomial Naive Bayes
    clf = MultinomialNB()
    clf.fit(count_data, y_train)

    test_count_data = count_vectorizer.transform(X_test).toarray()
    score = clf.score(test_count_data, y_test)

    print("Accuracy: {:.2f}%".format(score * 100))

    joblib.dump(count_vectorizer, '../pickle_model/features_model.pkl')
    joblib.dump(clf, '../pickle_model/predictive_model.pkl')


def read_category(category):
    filename_list = []
    txt_list = []
    txt_len_list = []

    file_list = os.listdir(category)
    for file in file_list:
        filename = os.path.join(category, file)
        txt = read_txt(filename)
        filename_list.append(filename.replace('/', '_'))
        txt_list.append(txt)
        txt_len_list.append(len(txt))

    dic = {
        'category': [category] * len(filename_list),
        'filename': filename_list,
        'raw_text': txt_list,
        'raw_text_lenght': txt_len_list
    }
    return pd.DataFrame.from_dict(dic)


def load_data(categories):
    data = pd.DataFrame()
    for category in categories:
        new_data = read_category(category)
        data = data.append(new_data)

    return data


def set_label(data, categories):
    cat_dic = dict(zip(categories, list(range(len(categories)))))
    data['category_label'] = data['category'].map(cat_dic)
    data = data[['category', 'filename', 'raw_text', 'text_4', 'category_label']]

    return data


train_classifier(sys.argv[1])

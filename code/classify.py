import sys
import joblib
import pandas as pd
from utils import read_txt, preprocess_data


def classify_documents():
    # load models
    model, features_model = load_models()

    # load data
    data = load_documents()

    # preprocess data
    data = preprocess_data(data)

    # create text features
    features = features_model.transform(data['text_4']).toarray()

    # classify documents
    data['scores'] = model.predict(features)

    print('\nDocument classification:\n')
    for filename, topic in zip(data['filename'], data['scores']):
        print(' - Document {} belongs to category {}'.format(filename, topic))


def load_models():
    features_model = joblib.load('../../pickle_model/features_model.pkl')
    model = joblib.load(sys.argv[1])
    return model, features_model


def load_documents():
    txt_list = []
    for document in sys.argv[2:]:
        txt = read_txt(document)
        txt_list.append(txt)

    dic = {
        'filename': sys.argv[2:],
        'raw_text': txt_list
    }
    return pd.DataFrame.from_dict(dic)


classify_documents()

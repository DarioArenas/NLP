import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def read_txt(filename):
    file = open(filename, 'r', encoding='latin1')
    txt = file.read()

    return txt


def lemmatizer(data):
    # Downloading punkt and wordnet from NLTK
    nltk.download('punkt')
    nltk.download('wordnet')

    # Saving the lemmatizer into an object
    wordnet_lemmatizer = WordNetLemmatizer()

    nrows = len(data)
    lemmatized_text_list = []

    for row, fln in zip(range(0, nrows), data['filename']):

        # Create an empty list containing lemmatized words
        lemmatized_list = []

        # Save the text and its words into an object
        text = data.loc[data['filename'] == fln]['text_3'][row]
        text_words = text.split(" ")

        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

        # Join the list
        lemmatized_text = " ".join(lemmatized_list)

        # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)

    data['text_4'] = lemmatized_text_list

    return data


def drop_stopwords(data):
    # Downloading the stop words list
    nltk.download('stopwords')

    # Loading the stop words in english
    stop_words = list(stopwords.words('english'))

    add_stopwords = ['edu', 'com', 'article', 'write', 'get', 'would']
    stop_words = stop_words + add_stopwords

    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        data['text_4'] = data['text_4'].str.replace(regex_stopword, '')

    data['text_4'] = data['text_4'].apply(lambda text: re.sub(' +', ' ', text))

    return data


def preprocess_data(data):
    # Replace \r and \n
    data['text_1'] = data['raw_text'].str.replace("\r", " ")
    data['text_1'] = data['text_1'].str.replace("\n", " ")
    data['text_1'] = data['text_1'].str.replace("\t", " ")
    data['text_1'] = data['text_1'].str.strip()

    # " when quoting text
    data['text_1'] = data['text_1'].str.replace('"', '')

    # Lowercasing the text
    data['text_2'] = data['text_1'].str.lower()

    # puntuaction signs
    punctuation_signs = list("?:!.,;@--<>")
    data['text_3'] = data['text_2']

    for punct_sign in punctuation_signs:
        data['text_3'] = data['text_3'].str.replace(punct_sign, ' ')
    data['text_3'] = data['text_3'].str.strip()

    # lemmatize
    data = lemmatizer(data)
    # drop stopwords
    data = drop_stopwords(data)

    return data

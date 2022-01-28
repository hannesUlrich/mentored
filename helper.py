import numpy as np
import pandas as pd
import re
import spacy
import os
import time
import csv
import datetime
from collections import Counter


def get_tok():
    try:
        return spacy.load('de_core_news_md')
    except OSError:
        print('Downloading language model for the spaCy POS tagger\n'
              "(don't worry, this will only happen once)")
    from spacy.cli import download
    download('de_core_news_md')
    return spacy.load('de_core_news_md')


# tokenize the phrases, including lowercasing and Umlaute removal
def tokenize(tok, text):
    return [token.lemma_ for token in tok.tokenizer(filter_text(text))]


def filter_text(text):
    text = str(text)
    text = text.lower()
    text = umlaute(text)
    text = special_chars(text)
    text = filter_space(text)
    return text


def umlaute(input):
    return input.replace("ä", "ae").replace("Ä", "Äe").replace("ö", "oe").replace("Ö", "oe").replace("ü", "ue").replace(
        "Ü", "ue")


def special_chars(input):
    return input.replace("(", "").replace(")", "").replace("!", "").replace("[", "").replace("]", "").replace("/",
                                                                                                              "").replace(
        ",", "").replace(".", "")


def filter_space(input):
    return re.sub(' +', ' ', input)


def labelClasses(df):
    idxToCodes = {}
    i = 0
    for codes in df['code']:
        if codes not in idxToCodes.keys():
            idxToCodes[codes] = i
            i = i + 1
    return idxToCodes


def loadDataset(file):
    df = pd.read_csv(file, delimiter=';', index_col=None)
    df.columns = ['code', 'phrase']
    return df


def loadDataset_idx(file, word_idx, threshold):
    df = pd.read_csv(file, delimiter=';', index_col=None)
    df.columns = ['code', 'phrase']

    code_counter = Counter(df['code'])
    df['code_occurrence'] = df['code'].apply(lambda x: code_counter[x])
    # Remove Concepts with a low occurrence
    discard = df[df['code_occurrence'] < threshold]
    df = df[df['code_occurrence'] >= threshold]

    # Adding Phrase Length
    df['phrases_length'] = df['phrase'].apply(lambda x: len(str(x).split()))
    df = df[df['phrases_length'] > 0]

    # Classes indices
    idx = labelClasses(df)
    df['class'] = df['code'].apply(lambda x: idx[x])
    tok = get_tok()
    df['encoded'] = df['phrase'].apply(lambda x: np.array(encode_sentence(tok, x, word_idx)))
    return df, discard['code'].to_list()


def loadAugmentation(file, word_idx, discards):
    df = pd.read_csv(file, delimiter=';', index_col=None)
    df.columns = ['code', 'phrase']

    # remove the discarded concepts from the augmentation
    df = df[~df['code'].isin(discards)]

    code_counter = Counter(df['code'])
    df['code_occurrence'] = df['code'].apply(lambda x: code_counter[x])

    # Adding Phrase Length
    df['phrases_length'] = df['phrase'].apply(lambda x: len(str(x).split()))
    df = df[df['phrases_length'] > 0]

    # Classes indices
    idx = labelClasses(df)
    df['class'] = df['code'].apply(lambda x: idx[x])
    tok = get_tok()
    df['encoded'] = df['phrase'].apply(lambda x: np.array(encode_sentence(tok, x, word_idx)))
    return df


def encode_sentence(tok, text, vocab2index, N=10):
    tokenized = tokenize(tok, text)
    # Padding
    encoded = np.zeros(N, dtype=int)
    try:
        enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    except:
        print('')
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded


def encodeDataset(df):
    tok = get_tok()
    counts = Counter()
    for index, row in df.iterrows():
        counts.update(tokenize(tok, row['phrase']))

    for word in list(counts):
        if counts[word] < 2:
            del counts[word]

    vocab2index = {"": 0, "UNK": 1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)
    df['encoded'] = df['phrase'].apply(lambda x: np.array(encode_sentence(tok, x, vocab2index)))
    return df, words, vocab2index


def encodeDataset_with_index(idx, df):
    tok = get_tok()
    df['encoded'] = df['phrase'].apply(lambda x: np.array(encode_sentence(tok, x, idx)))
    return df


def construct_idx(df):
    tok = get_tok()
    counts = Counter()
    for index, row in df.iterrows():
        counts.update(tokenize(tok, row['phrase']))

    for word in list(counts):
        if counts[word] < 2:
            del counts[word]

    vocab2index = {"": 0, "UNK": 1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)

    with open('data/idx.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        for key, value in vocab2index.items():
            writer.writerow([key, value])


def load_idx(file):
    with open(file, mode='r') as infile:
        reader = csv.reader(infile, delimiter=';')
        idx = dict((rows[0], rows[1]) for rows in reader)
    return idx

def mk_result_dir(dir_path):
    try:
        os.mkdir(dir_path)
        return dir_path
    except:
        new_path = dir_path + "_" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
        new_path = mk_result_dir(new_path)
        return new_path
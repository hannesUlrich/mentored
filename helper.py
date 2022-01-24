import numpy as np
import pandas as pd
import spacy
import os
import time
import datetime

from collections import Counter

import umls_connect


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
    return [token.text for token in tok.tokenizer(umlaute(str(text).lower()))]


def umlaute(input):
    return input.replace("ä","ae").replace("Ä","Äe").replace("ö","oe").replace("Ö","oe").replace("ü","ue").replace("Ü","ue")


def encode_sentence(tok, text, vocab2index, N=10):
    tokenized = tokenize(tok, text)
    # Padding
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded


def encodeDataset(df):
    tok = get_tok()
    counts = Counter()
    for index, row in df.iterrows():
        counts.update(tokenize(tok, row['phrase']))

    print("num_words before:", len(counts.keys()))
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]
    print("num_words after:", len(counts.keys()))

    vocab2index = {"": 0, "UNK": 1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)
    df['encoded'] = df['phrase'].apply(lambda x: np.array(encode_sentence(tok, x, vocab2index)))
    return df, words, vocab2index


def mk_result_dir(dir_path):
    try:
        os.mkdir(dir_path)
        return dir_path
    except:
        new_path = dir_path+"_"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
        new_path = mk_result_dir(new_path)
        return new_path
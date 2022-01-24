import pandas as pd
import umls_connect
import argparse
from collections import Counter


def pickled(file, threshold):
    df = pd.read_csv(file, delimiter=';', index_col=None)
    df.columns = ['code', 'phrase']

    # Remove invalid Concepts
    df = df[df['code'].str.startswith('C')]

    # Adding Phrase Length
    df['phrases_length'] = df['phrase'].apply(lambda x: len(str(x).split()))

    # Remove Concepts with a low occurrence
    code_counter = Counter(df['code'])
    df['code_occurrence'] = df['code'].apply(lambda x: code_counter[x])
    df = df[df['code_occurrence'] > threshold]

    # Classes indices
    idx = labelClasses(df)
    df['class'] = df['code'].apply(lambda x: idx[x])

    # Adding Displays and SemanticTypes
    displays = []
    types = []

    cache = {}
    for index, row in df.iterrows():
        if row['code'] in cache.keys():
            displays.append(cache[row['code']][0])
            types.append(''.join(cache[row['code']][1]))
        else:
            display, type = umls_connect.getDisplayAndType(row['code'])
            displays.append(display.strip(',;"'))
            types.append(''.join(type))
            cache[row['code']] = [display, type]
    print(len(cache.keys()))
    df['display'] = displays
    df['types'] = types

    # save class and corresponding codes, displays and types
    temp = df[['class', 'code', 'display', 'types']]
    temp = temp.drop_duplicates(subset=['class'])
    temp.to_csv(f'data/displays.csv')

    return df


def labelClasses(df):
    idxToCodes = {}
    i = 0
    for codes in df['code']:
        if codes not in idxToCodes.keys():
            idxToCodes[codes] = i
            i = i + 1
    return idxToCodes

# The function excepts a CSV File with the following shape:
#  "CODE";"PHRASE" --> "C0027989;newspapers"
#   The Threshold specifies the number of samples that must be present for a code.
#   If there are less, the code will be removed from the training dataset.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process')
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--threshold', type=int, default=0)
    args = parser.parse_args()
    pickled_data = pickled(args.file, args.threshold)
    pickled_data.to_pickle("data/data.pkl")
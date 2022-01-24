import pandas as pd
import time
import csv
import construct_pickle
import umls_connect


def augment_pickle():
    pickled_phrases = pd.read_pickle("data/data.pkl")
    codes = pickled_phrases['code_occurrence'].unique()

    mesh_phrases = list()
    for index, row in codes.iterrows():
        code = row['code']
        mesh = umls_connect.queryMeSH(code)
        for m in mesh:
            mesh_phrases.append(f'{code};{m}')

    with open("data/augment_mesh.csv", 'w', newline='\n') as destinationFile:
        writer = csv.writer(destinationFile, delimiter=';',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(mesh_phrases)

    aug_pickle = construct_pickle.pickled("data/augment_mesh.csv", threshold=0)
    aug_pickle.to_pickle('data/aug.pkl')

if __name__ == '__main__':
    tic = time.perf_counter()
    augment_pickle()
    toc = time.perf_counter()
    print(f"{toc - tic:0.4f} seconds")



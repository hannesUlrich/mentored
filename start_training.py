import pandas as pd
import argparse
import Network
import ds
import helper
import os
from collections import Counter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import test_model
from train import train_model


def assemble(name, epoch, threshold, device):

    mdm_path = 'data/XXXXX.csv'
    mesh_path = 'data/XXXXX.csv'

    if not os.path.isfile('data/idx.csv'):
        print ("No Word Index found, start constructing...")
        mdm = helper.loadDataset(mdm_path)
        mesh = helper.loadDataset(mesh_path)
        combined = pd.concat([mdm, mesh])
        helper.construct_idx(combined)

    print ("Loading Word Index!")
    idx = helper.load_idx('data/idx.csv')

    # Load data without the discarded concepts
    mdm, discards = helper.loadDataset_idx(mdm_path, word_idx=idx, threshold=threshold)
    mesh = helper.loadAugmentation(mesh_path, word_idx=idx, discards=discards)

    # Make results directory
    dir_path = f'results/{name}'
    dir_path = helper.mk_result_dir(dir_path)

    # Split encoded dataset into test and training
    train, test = train_test_split(mdm, test_size=0.2)
    test.to_csv(f'{dir_path}/test.csv', sep=';')

    # Combine splitted Trainings Data with Augmentation
    combined = pd.concat([train, mesh])
    combined.drop_duplicates(subset=['code', 'phrase'])

    # split data + aug --> train / eval 116854
    X = list(combined['encoded'])
    y = list(combined['class'])
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.05)
    train_ds = ds.data(X_train, y_train)
    valid_ds = ds.data(X_valid, y_valid)

    # train
    batch_size = 5000
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=batch_size)

    # len(idx.keys()) = 38471
    vocab_size = 39000
    embedding_dim = 128
    hidden_dim = 32
    num_class = len(Counter(mdm['class']).keys())
    num_layers = 2

    print("Building the model")
    model = Network.classifier(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                               output_dim=num_class, n_layers=num_layers,
                               bidirectional=True, dropout=0)

    print('Start training')
    bst_acc = train_model(device, dir_path, model, train_dl, val_dl, epochs=int(epoch), lr=0.01)
    print(f'Training complete with Accuracy {bst_acc} and {num_class} classes.')
    test_model.test(dir_path, 'cpu', True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--epoch', default=150, type=str)
    parser.add_argument('--threshold', default=10, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    args = parser.parse_args()

    assemble(args.name, args.epoch, args.threshold, args.device)

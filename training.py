import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import DataLoader
import Classifier
import helper as hp
import training_ds


def createModel(name, epoch=250, device='cpu'):

    pickled_phrases = pd.read_pickle("data/data.pkl")
    aug_phrases = pd.read_pickle("data/aug.pkl")
    combined_df = pd.concat(pickled_phrases, aug_phrases)

    phrases_encoded, words, idx = hp.encodeDataset(combined_df)
    num_class = len(Counter(phrases_encoded['class']).keys())

    # Make results directory
    dir_path = f'results/{name}'
    dir_path = hp.mk_result_dir(dir_path)

    # save word index
    idx_df = pd.DataFrame.from_dict(idx, orient='index')
    idx_df.to_csv(f'{dir_path}/words.csv')

    # Split encoded dataset into test and training
    train_encoded, test = train_test_split(phrases_encoded, test_size=0.2)
    test.to_csv(f'{dir_path}/test.csv')

    X = list(train_encoded['encoded'])
    y = list(train_encoded['class'])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    train_ds = training_ds.Data(X_train, y_train)
    valid_ds = training_ds.Data(X_valid, y_valid)

    batch_size = 5000
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=batch_size)

    vocab_size = len(words)
    embedding_dim = 128
    hidden_dim = 32
    num_class = len(Counter(phrases_encoded['class']).keys())
    print(num_class)
    num_layers = 4

    print("Building the model")
    model = Classifier.Model(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                             output_dim=num_class, n_layers=num_layers,
                             bidirectional=True, dropout=0.3)

    print('Start training')
    bst_acc = train_model(device, dir_path, model, train_dl, val_dl, epochs=int(epoch), lr=0.01)
    print(f'Training complete with Accuracy {bst_acc}')


def train_model(device, dir_path, model, train_dl, val_dl, epochs, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    model.to(device)

    best_acc = 0.0

    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, x_len in train_dl:
            x = x.long().to(device)
            y = y.long().to(device)
            y_pred = model(x, x_len)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y, reduction='none')
            loss = torch.mean(loss * x_len.to(device) / 10)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, val_rmse = validation_metrics(device, model, val_dl)

        if best_acc <= val_acc:
            best_acc = val_acc
            torch.save(model, f'{dir_path}/best_model.pt')
        print("Epoch %d: train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (
            (i + 1), sum_loss / total, val_loss, val_acc, val_rmse))
    return best_acc


def validation_metrics(device, model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y, x_len in valid_dl:
        x = x.long().to(device)
        y = y.long().to(device)
        y_hat = model(x, x_len).to(device)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred.cpu(), y.unsqueeze(-1).cpu())) * y.shape[0]
    return sum_loss / total, correct / total, sum_rmse / total

if __name__ == '__main__':

    name = "mentored"
    epoch = 250
    device = "cpu"

    createModel(name, epoch, device)

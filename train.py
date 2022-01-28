wimport torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error


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
            (i+1), sum_loss / total, val_loss, val_acc, val_rmse))
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
        ## TODO change to torch.MSELoss
        sum_rmse += np.sqrt(mean_squared_error(pred.cpu(), y.unsqueeze(-1).cpu())) * y.shape[0]
    return sum_loss / total, correct / total, sum_rmse / total
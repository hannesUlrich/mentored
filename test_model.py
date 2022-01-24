import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from texttable import Texttable
import argparse

import training_ds


def printTopTable(t1, t3, t5, t10, total):
    t1_hit_ratio = format((t1['hit'] / total * 100), '.2f')
    t1_miss_ratio = format((t1['miss'] / total * 100), '.2f')
    t3_hit_ratio = format((t3['hit'] / total * 100), '.2f')
    t3_miss_ratio = format((t3['miss'] / total * 100), '.2f')
    t5_hit_ratio = format((t5['hit'] / total * 100), '.2f')
    t5_miss_ratio = format((t5['miss'] / total * 100), '.2f')
    t10_hit_ratio = format((t10['hit'] / total * 100), '.2f')
    t10_miss_ratio = format((t10['miss'] / total * 100), '.2f')

    table = Texttable()
    table.add_row(['T#', 'Hit', 'Miss', 'Hit %', 'Miss %', 'total'])
    table.add_row(['T1', t1['hit'], t1['miss'], t1_hit_ratio, t1_miss_ratio, total])
    table.add_row(['T3', t3['hit'], t3['miss'], t3_hit_ratio, t3_miss_ratio, total])
    table.add_row(['T5', t5['hit'], t5['miss'], t5_hit_ratio, t5_miss_ratio, total])
    table.add_row(['T10', t10['hit'], t10['miss'], t10_hit_ratio, t10_miss_ratio, total])
    print(table.draw())


def test_model(path, device):
    model_file = path + '/model.pt'
    test_file = path + '/test.csv'

    model = torch.load(model_file, map_location=torch.device('cpu')).to(device)
    model.eval()

    test = pd.read_csv(test_file)

    ## Preprocess the model
    x_test = list(test['encoded'])
    y_test = list(test['class'])
    x_test_casted = list()
    for x in x_test:
        x_test_casted.append(np.array([int(s) for s in x.strip('[]').split() if s.isdigit()]))
    test_ds = training_ds.Data(x_test_casted, y_test)

    ## DataLoader
    batch_size = 1
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    t1_dict = {'hit': 0, 'miss': 0}
    t3_dict = {'hit': 0, 'miss': 0}
    t5_dict = {'hit': 0, 'miss': 0}
    t10_dict = {'hit': 0, 'miss': 0}

    with torch.no_grad():
        for x, y, x_len in test_dl:
            x = x.long().to(device)
            y_pred = model(x, x_len)

            # Top k Predictions
            t1 = torch.topk(y_pred, 1)
            t3 = torch.topk(y_pred, 3)
            t5 = torch.topk(y_pred, 5)
            t10 = torch.topk(y_pred, 10)

            # Count if the code is within the T3 or not
            if y.item() in t1[1][0].numpy():
                t1_dict['hit'] = t1_dict['hit'] + 1
            else:
                t1_dict['miss'] = t1_dict['miss'] + 1

            # Count if the code is within the T3 or not
            if y.item() in t3[1][0].numpy():
                t3_dict['hit'] = t3_dict['hit'] + 1
            else:
                t3_dict['miss'] = t3_dict['miss'] + 1

            # Count if the code is within the T5 or not
            if y.item() in t5[1][0].numpy():
                t5_dict['hit'] = t5_dict['hit'] + 1
            else:
                t5_dict['miss'] = t5_dict['miss'] + 1

            # Count if the code is within the T10 or not
            if y.item() in t10[1][0].numpy():
                t10_dict['hit'] = t10_dict['hit'] + 1
            else:
                t10_dict['miss'] = t10_dict['miss'] + 1

    print(str(model_file))
    printTopTable(t1_dict, t3_dict, t5_dict, t10_dict, len(test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--device', default='cpu', type=str)
    args = parser.parse_args()
    test_model(args.path, args.device)

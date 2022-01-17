from utils import getData, encoder, createLabel
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os


def read_src():
    path = 'data/src'
    filelist = os.listdir(path)
    files = []
    for f in filelist:
        file = os.path.join(path, f)
        files.append(file)
    return files


def load_train(max_size=100, isencoder=True):
    files = read_src()
    src = []
    for f in files:
        pos = os.path.join(f, 'pos.txt')
        neg = os.path.join(f, 'neg.txt')
        print(pos, neg)
        posseqs = getData(pos)
        negseqs = getData(neg)

        if isencoder:
            posdata = encoder(posseqs, max_size)
            negdata = encoder(negseqs, max_size)
        else:
            posdata = np.array(posseqs)
            negdata = np.array(negseqs)

        data = posdata.tolist() + negdata.tolist()
        data = np.array(data)
        label = createLabel(len(posseqs), len(negseqs))
        src.append([data, label])
    return src


def load_train_test(max_size=100, split=True, isencoder=True):  # target
    posseqs = getData('data/tgt/after_catch_promoters.txt')
    negseqs = getData('data/tgt/non_promoters.txt')

    if isencoder:
        posdata = encoder(posseqs, max_size)
        negdata = encoder(negseqs, max_size)
    else:
        posdata = np.array(posseqs)
        negdata = np.array(negseqs)

    if split:
        train_pos, test_pos = train_test_split(posdata, test_size=0.2, shuffle=False)
        train_neg, test_neg = train_test_split(negdata, test_size=0.2, shuffle=False)

        train_data = train_pos.tolist() + train_neg.tolist()
        train_data = np.array(train_data)
        train_label = createLabel(len(train_pos), len(train_neg))

        test_data = test_pos.tolist() + test_neg.tolist()
        test_data = np.array(test_data)
        test_label = createLabel(len(test_pos), len(test_neg))

        return [train_data, train_label], [test_data, test_label]
    else:
        data = posdata.tolist() + negdata.tolist()
        data = np.array(data)
        label = createLabel(len(posseqs), len(negseqs))

        return [data, label]

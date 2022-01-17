from model import Transfer_fea_MT
from dataloader import load_train, load_train_test
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from evaluation import scores
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import numpy as np
import random
from keras.models import load_model
import tensorflow as tf
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# session = tf.Session(config=config)


def pair_xfy(src_data, src_fea, src_label, tgt_data, tgt_fea, tgt_label):
    len_tgt = len(tgt_data)
    list_len_tgt = list(range(0, len_tgt))
    random.shuffle(list_len_tgt)

    Training_P = []
    Training_N = []

    trt_index = 0
    for src_index in range(len(src_data)):
        if src_label[src_index] == tgt_label[list_len_tgt[trt_index]]:
            Training_P.append([src_index, list_len_tgt[trt_index]])
        else:
            Training_N.append([src_index, list_len_tgt[trt_index]])
        trt_index = trt_index + 1
        if trt_index >= len_tgt:
            trt_index = 0

    Training = Training_P + Training_N
    random.shuffle(Training)

    X1, X2 = [], []
    F1, F2 = [], []
    y1, y2, yc = [], [], []
    for i in range(len(Training)):
        index1, index2 = Training[i]
        X1.append(src_data[index1])
        X2.append(tgt_data[index2])
        F1.append(src_fea[index1])
        F2.append(tgt_fea[index2])
        y1.append(src_label[index1])
        y2.append(tgt_label[index2])
        if src_label[index1] != tgt_label[index2]:
            yc.append(1)
        else:
            yc.append(0)
    return np.array(X1), np.array(F1), np.array(y1), np.array(X2), np.array(F2), np.array(y2), np.array(yc)


def kfold(src, tgt, fold_num=10, weight=0.2):
    src_data, src_fea, src_label = src[0], src[1], src[2]
    tgt_X, tgt_F, tgt_y = tgt[0], tgt[1], tgt[2]

    Recall, SPE, Precision, F1, MCC, Acc, AUC, AUPR = [], [], [], [], [], [], [], []

    skf = StratifiedKFold(n_splits=fold_num, shuffle=True)
    for train_index, test_index in skf.split(tgt_X, tgt_y):
        tgt_data_train, tgt_data_test = tgt_X[train_index], tgt_X[test_index]
        tgt_fea_train, tgt_fea_test = tgt_F[train_index], tgt_F[test_index]
        tgt_label_train, tgt_label_test = tgt_y[train_index], tgt_y[test_index]

        s_X, s_F, s_y, t_X, t_F, t_y, yc = pair_xfy(src_data, src_fea, src_label, tgt_data_train, tgt_fea_train,
                                                    tgt_label_train)

        model = Transfer_fea_MT(99, 7, weight)
        model.fit([t_X, t_F, s_X, s_F], [t_y, s_y], epochs=10, verbose=2, shuffle=True)
        y_pred, _ = model.predict([tgt_data_test, tgt_fea_test, tgt_data_test, tgt_fea_test])
        score = scores(tgt_label_test, y_pred)

        Recall.append(score[0])
        SPE.append(score[1])
        Precision.append(score[2])
        F1.append(score[3])
        MCC.append(score[4])
        Acc.append(score[5])
        AUC.append(score[6])
        AUPR.append(score[7])
    print("Transfer Recall:%.4f, SPE:%.4f, Precision:%.4f, F1:%.4f, MCC:%.4f, ACC:%.4f, AUC:%.4f, AUPR:%.4f" %
          (np.mean(Recall), np.mean(SPE), np.mean(Precision), np.mean(F1), np.mean(MCC), np.mean(Acc), np.mean(AUC),
           np.mean(AUPR)))
    return [np.mean(Recall), np.mean(SPE), np.mean(Precision), np.mean(F1), np.mean(MCC), np.mean(Acc), np.mean(AUC),
            np.mean(AUPR)]


def mt_train(model, src, tgt):
    src_data, src_fea, src_label = src[0], src[1], src[2]
    tgt_X, tgt_F, tgt_y = tgt[0], tgt[1], tgt[2]

    s_X, s_F, s_y, t_X, t_F, t_y, yc = pair_xfy(src_data, src_fea, src_label, tgt_X, tgt_F, tgt_y)
    model.fit([t_X, t_F, s_X, s_F], [t_y, s_y], epochs=15, verbose=2, shuffle=True)
    model.save('./model/transfer/transfer.h5')


if __name__ == '__main__':
    in_length = 99
    source_data = load_train(max_size=in_length, isencoder=True)
    data = []
    label = []
    for i in range(len(source_data)):
        data = data + source_data[i][0].tolist()
        label = label + source_data[i][1].tolist()
    src_X = np.array(data)
    src_y = np.array(label)
    source_data = load_train(isencoder=False)
    data = []
    for i in range(len(source_data)):
        data = data + source_data[i][0].tolist()
    from feature import com_seq_feature

    src_f = com_seq_feature(data)
    src = [src_X, src_f, src_y]

    tgt = load_train_test(split=False, max_size=in_length, isencoder=True)
    t_ = load_train_test(isencoder=False, split=False)
    tgt_f = com_seq_feature(t_[0])
    tgt = [tgt[0], tgt_f, tgt[1]]

    Recall, SPE, Precision, F1, MCC, Acc, AUC, AUPR = [], [], [], [], [], [], [], []
    for i in range(5):
        score = kfold(src, tgt, weight=0.2)

        Recall.append(score[0])
        SPE.append(score[1])
        Precision.append(score[2])
        F1.append(score[3])
        MCC.append(score[4])
        Acc.append(score[5])
        AUC.append(score[6])
        AUPR.append(score[7])
    print("Transfer Recall:%.4f, SPE:%.4f, Precision:%.4f, F1:%.4f, MCC:%.4f, ACC:%.4f, AUC:%.4f, AUPR:%.4f" %
          (np.mean(Recall), np.mean(SPE), np.mean(Precision), np.mean(F1), np.mean(MCC), np.mean(Acc), np.mean(AUC),
           np.mean(AUPR)))

    # from val_ind import data_process, val
    # model = Transfer_fea_MT(99, 7, 0.2)
    # mt_train(model, src, tgt)
    #
    # data_path1 = './data/ind/independ_test_promoters1.txt'
    # data_path2 = './data/ind/independ_test_promoters2.txt'
    # data_path3 = './data/ind/independ_test_promoters3.txt'
    #
    # neg_path1 = './data/ind/independent_neg1.txt'
    # neg_path2 = './data/ind/independent_neg2.txt'
    # neg_path3 = './data/ind/independent_neg3.txt'
    #
    # valset = [[data_path1, neg_path1], [data_path2, neg_path2], [data_path3, neg_path3]]
    #
    # for v in valset:
    #     data, fea, label = data_process(v[0], v[1], max_len=99)
    #     val(model, data, fea, label)

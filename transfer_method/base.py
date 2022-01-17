from model import base_feature
from dataloader import load_train, load_train_test
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from evaluation import scores
from sklearn.model_selection import StratifiedKFold
import numpy as np
from keras.optimizers import Adam
from feature import com_seq_feature
import os
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# session = tf.Session(config=config)


def kfold(data, fold_num=10):
    X, F, y = data[0], data[1], data[2]
    Recall, SPE, Precision, F1, MCC, Acc, AUC, AUPR = [], [], [], [], [], [], [], []
    skf = StratifiedKFold(n_splits=fold_num, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        print("fold:" + str(i))
        trainX, testX = X[train_index], X[test_index]
        trainF, testF = F[train_index], F[test_index]
        trainY, testY = y[train_index], y[test_index]

        model = base_feature()

        model.fit([trainX, trainF], trainY, epochs=150, batch_size=64, verbose=2, shuffle=True)
        y_pred = model.predict([testX, testF])
        score = scores(testY, y_pred)

        Recall.append(score[0])
        SPE.append(score[1])
        Precision.append(score[2])
        F1.append(score[3])
        MCC.append(score[4])
        Acc.append(score[5])
        AUC.append(score[6])
        AUPR.append(score[7])
    print("Base Recall:%.4f, SPE:%.4f, Precision:%.4f, F1:%.4f, MCC:%.4f, ACC:%.4f, AUC:%.4f, AUPR:%.4f" %
          (np.mean(Recall), np.mean(SPE), np.mean(Precision), np.mean(F1), np.mean(MCC), np.mean(Acc), np.mean(AUC),
          np.mean(AUPR)))
    return [np.mean(Recall), np.mean(SPE), np.mean(Precision), np.mean(F1), np.mean(MCC), np.mean(Acc), np.mean(AUC), np.mean(AUPR)]
    

def train(model, data, epochs):
    X, F, y = data[0], data[1], data[2]

    model.fit([X, F], y, epochs=epochs, batch_size=64, verbose=2, shuffle=True)
    model.save('./model/base/base.h5')


if __name__ == '__main__':
    epoch_num = 150
    in_length = 99
    out_length = 1

    tgt = load_train_test(max_size=99, split=False)
    t_ = load_train_test(isencoder=False, split=False, max_size=in_length)
    tgt_f = com_seq_feature(t_[0])
    
    Recall, SPE, Precision, F1, MCC, Acc, AUC, AUPR = [], [], [], [], [], [], [], []
    for i in range(5):

        score = kfold([tgt[0], tgt_f, tgt[1]])

        Recall.append(score[0])
        SPE.append(score[1])
        Precision.append(score[2])
        F1.append(score[3])
        MCC.append(score[4])
        Acc.append(score[5])
        AUC.append(score[6])
        AUPR.append(score[7])
    print("TargetOnly Recall:%.4f, SPE:%.4f, Precision:%.4f, F1:%.4f, MCC:%.4f, ACC:%.4f, AUC:%.4f, AUPR:%.4f" %
          (np.mean(Recall), np.mean(SPE), np.mean(Precision), np.mean(F1), np.mean(MCC), np.mean(Acc), np.mean(AUC),
           np.mean(AUPR)))


    # model = base_feature()
    # train(model, [tgt[0], tgt_f, tgt[1]], epoch_num)
    #
    # from val_ind import data_process, val_cnn
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
    #     val_cnn(model, data, fea, label)

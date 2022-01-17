from utils import getData, encoder, createLabel
from evaluation import scores
import numpy as np
from feature import com_seq_feature


def data_process(pos_path, neg_path, max_len=99):
    posseqs = getData(pos_path)
    negseqs = getData(neg_path)
    posdata = encoder(posseqs, max_len)
    negdata = encoder(negseqs, max_len)
    posfea = com_seq_feature(posseqs)
    negfea = com_seq_feature(negseqs)

    data = posdata.tolist() + negdata.tolist()
    data = np.array(data)
    fea = posfea.tolist() + negfea.tolist()
    fea = np.array(fea)
    label = createLabel(len(posseqs), len(negseqs))
    return data, fea, label


def val_cnn(model, data, fea, y_true):
    y_pred = model.predict([data, fea])
    score = scores(y_true, y_pred, th=0.5)
    print("VAL Recall:%.4f, SPE:%.4f, Precision:%.4f, F1:%.4f, MCC:%.4f, ACC:%.4f, AUC:%.4f, AUPR:%.4f" %
          (score[0], score[1], score[2], score[3], score[4], score[5], score[6], score[7]))


def val(model, data, fea, y_true):
    y_pred, _ = model.predict([data, fea, data, fea])
    score = scores(y_true, y_pred, th=0.5)
    print("VAL Recall:%.4f, SPE:%.4f, Precision:%.4f, F1:%.4f, MCC:%.4f, ACC:%.4f, AUC:%.4f, AUPR:%.4f" %
          (score[0], score[1], score[2], score[3], score[4], score[5], score[6], score[7]))

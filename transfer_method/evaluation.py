from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, auc
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve


def scores(y_test, y_pred, th=0.5):
    y_predlabel = [(0 if item < th else 1) for item in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, y_predlabel).flatten()
    SPE = tn * 1. / (tn + fp)
    MCC = matthews_corrcoef(y_test, y_predlabel)
    Recall = recall_score(y_test, y_predlabel)
    Precision = precision_score(y_test, y_predlabel)
    F1 = f1_score(y_test, y_predlabel)
    Acc = accuracy_score(y_test, y_predlabel)
    AUC = roc_auc_score(y_test, y_pred)
    precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)
    AUPR = auc(recall_aupr, precision_aupr)
    return [Recall, SPE, Precision, F1, MCC, Acc, AUC, AUPR, tp, fn, tn, fp]

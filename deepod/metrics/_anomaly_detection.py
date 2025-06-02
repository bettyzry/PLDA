from sklearn import metrics
import numpy as np


def auc_roc(y_true, y_score):
    return metrics.roc_auc_score(y_true, y_score)


def auc_pr(y_true, y_score):
    return metrics.average_precision_score(y_true, y_score)


def tabular_metrics(y_true, y_score):
    """calculate evaluation metrics"""

    # F1@k, using real percentage to calculate F1-score
    ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
    thresh = np.percentile(y_score, ratio)
    y_pred = (y_score >= thresh).astype(int)
    y_true = y_true.astype(int)
    p, r, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return auc_roc(y_true, y_score), auc_pr(y_true, y_score), f1


def ts_metrics(y_true, y_score):
    """calculate evaluation metrics for time series anomaly detection"""
    best_f1, best_p, best_r = get_best_f1(y_true, y_score)
    return auc_roc(y_true, y_score), auc_pr(y_true, y_score), best_f1, best_p, best_r


def get_best_f1(label, score):
    precision, recall, _ = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    return best_f1, best_p, best_r


def DC_metrics(y_true, y_score, rate=0.5):
    thresh = np.percentile(y_score, 100 - rate)
    pred = (y_score > thresh).astype(int)

    gt = y_true.astype(int)

    print("pred:   ", pred.shape)
    print("gt:     ", gt.shape)

    # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred = np.array(pred)
    gt = np.array(gt)
    print("pred: ", pred.shape)
    print("gt:   ", gt.shape)

    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary')
    print(
        "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
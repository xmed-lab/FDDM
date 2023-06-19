from sklearn import metrics


def f1_score(A, B):
    if A == 0 and B == 0:
        return 0
    return 2.0 * A * B / (A + B)


def accuracy_score(conf_mat):
    return float(conf_mat[0][0] + conf_mat[1][1]) / float(
        conf_mat[0][1] + conf_mat[0][0] + conf_mat[1][0] + conf_mat[1][1])


def confusion_matrix(expects, predicts):
    return metrics.confusion_matrix(expects, predicts)


def multilabel_confusion_matrix(expects, predicts):
    return metrics.multilabel_confusion_matrix(expects, predicts)


def spe_score(confusion_matrix):
    if confusion_matrix[0][0] == 0 and confusion_matrix[1][0] == 0:
        return 0.0
    return float(confusion_matrix[0][0]) / float(confusion_matrix[1][0] + confusion_matrix[0][0])


def sen_score(confusion_matrix):
    if confusion_matrix[0][1] == 0 and confusion_matrix[1][1] == 0:
        return 0.0
    return float(confusion_matrix[1][1]) / float(confusion_matrix[1][1] + confusion_matrix[0][1])

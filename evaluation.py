import numpy as np
from tqdm import tqdm 
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


def prfga(y_pred, y_test): 
    f, p, r, g, a = 0, 0, 0, 0, 0 
    y_t = np.array([[1 if y_pred[i,j]>=0.5 else 0 for j in range(y_pred.shape[1])] for i in range(len(y_pred))]) 
    for i in tqdm(range(y_t.shape[0])): 
        ps, rc, f1, _ = precision_recall_fscore_support(y_test[i], y_t[i], average='macro')
        acc = accuracy_score(y_test[i], y_t[i])
        gm = math.sqrt(ps*rc)
        p += ps 
        r += rc
        f += f1 
        g += gm
        a += acc

    return p/y_test.shape[0], r/y_test.shape[0], f/y_test.shape[0], g/y_test.shape[0], a/y_test.shape[0]


def prfga_(y_pred, y_test): 
    f, p, r, g, a = 0, 0, 0, 0, 0 
    y_t = np.array([[1 if y_pred[i,j]>=0.5 else 0 for j in range(y_pred.shape[1])] for i in range(len(y_pred))]) 
    for i in tqdm(range(y_t.shape[1])): 
        ps, rc, f1, _ = precision_recall_fscore_support(y_test[:, i], y_t[:, i], average='macro')
        acc = accuracy_score(y_test[:, i], y_t[:, i])
        gm = math.sqrt(ps*rc)
        p += ps 
        r += rc
        f += f1 
        g += gm
        a += acc
    
    return p/y_test.shape[1], r/y_test.shape[1], f/y_test.shape[1], g/y_test.shape[1], a/y_test.shape[1]


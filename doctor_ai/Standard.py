import numpy as np


def precision_top(y_true, y_predict, rank=None):
    if rank is None:
        rank = [1, 2, 3, 4, 5,6,7,8,9,10]
    pre = list()
    for i in range(len(y_predict)):
        this_one = list()
        count = 0
        for j in y_true[i]:
            if j == 1:
                count += 1
        if count:
            codes = np.argsort(y_true[i])
            tops = np.argsort(y_predict[i])
            for rk in rank:
                if len(set(codes[len(codes) - count:]).intersection(set(tops[len(tops) - rk:]))) >= 1:
                    this_one.append(1)
                else:
                    this_one.append(0)
            pre.append(this_one)
    return (np.array(pre)).mean(axis=0).tolist()

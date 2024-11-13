import numpy as np

### 定义二叉特征分裂函数
def feature_split(X, feature_i, threshold):
    '''
    
    '''
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    X_left = np.array([sample for sample in X if split_func(sample)])
    X_right = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_left, X_right])


# 基尼系数
def calculate_gini(y):
    '''
    计算一组数据的基尼系数来衡量数据集纯度，系数越小，纯度越高
    公式为: sum(p*(1-p)) = 1 - sum(p^2) 因为sum(p) = 1
    '''
    y = y.tolist()
    probs = [y.count(i)/len(y) for i in np.unique(y)]
    gini = 1-sum([p**2 for p in probs])
    # gini = sum([p*(1-p) for p in probs])
    return gini


# 打乱数据
def data_shuffle(X,y,seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


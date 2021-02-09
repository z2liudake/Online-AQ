import numpy as np

def getRecallFunc(T, result, gt):
    recall = 0.0
    for i in range(result.shape[0]):
        if gt[i] in result[i,:T]:
            recall += 1
    return recall / result.shape[0]

def getRecallFuncBatch(T, result, gt):
    recall = 0.0
    for i in range(result.shape[0]):
        if gt[i] in result[i,:T]:
            recall += 1
    return recall

def getRecall(querySource, baseSource, result, threadCounts=10):
    print(f'step into getRecall function.')
    print(f'querySource shape is {querySource.shape}.')
    print(f'baseSource shape is {baseSource.shape}')

    q_2 = np.linalg.norm(querySource, axis=1).reshape(-1,1) ** 2
    x_2 = np.linalg.norm(baseSource, axis=1).reshape(1,-1) ** 2

    q_x = np.dot(querySource, baseSource.T)

    gt_distance = q_2 - 2*q_x + x_2

    gt = np.argmin(gt_distance, axis=1)

    T = [2**i for i in range(11)]
    T.insert(5, 20)

    res = np.zeros(len(T))

    for i, t in enumerate(T):
        length = min(t, baseSource.shape[0])
        cur_value = getRecallFunc(length, result, gt)
        print (f'Recall@{length} {cur_value}')
        res[i] = cur_value
    return res

def getRecallBatch(querySource, baseSource, result, threadCounts=10):

    q_2 = np.linalg.norm(querySource, axis=1).reshape(-1,1) ** 2
    x_2 = np.linalg.norm(baseSource, axis=1).reshape(1,-1) ** 2

    q_x = np.dot(querySource, baseSource.T)

    gt_distance = q_2 - 2*q_x + x_2

    gt = np.argmin(gt_distance, axis=1)

    T = [2**i for i in range(11)]
    T.insert(5, 20)

    res = np.zeros(len(T))

    for i, t in enumerate(T):
        length = min(t, baseSource.shape[0])
        cur_value = getRecallFuncBatch(length, result, gt)
        res[i] = cur_value
    return res


def getRecallStreaming(query, baseSource, result, maxK):

    q_2 = np.linalg.norm(query) ** 2
    x_2 = np.linalg.norm(baseSource, axis=1).reshape(-1) ** 2
    q_x = np.dot(baseSource, query)    

    gt_distance = q_2 - 2*q_x + x_2

    gt = np.argmin(gt_distance)
    
    T = [2**i for i in range(maxK)]
    T.insert(5, 20)

    res = np.zeros(len(T))

    for i, t in enumerate(T):
        length = min(t, baseSource.shape[0])
        if gt in result[:t]:
            res[i] += 1 
            
    return res
def getRecallAt(length, result, groundtruth):
    recall = 0.0
    for i in range(result.shape[0]):
        if groundtruth[i,0] in result[i,:length]:
            recall += 1
    return recall / result.shape[0]


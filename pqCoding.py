import time
import pickle
from functools import partial
from multiprocessing import Pool
from collections import deque

import numpy as np
import pandas as pd
from scipy import io as matIO
from scipy.cluster.vq import vq, kmeans2

from utils import *
from newIO import *
from getRecallOnline import getRecall

winSize = 1000

def runKmeans2(m, vocabDim, points, K, itsCount, minit):
    subPoints = points[:, m * vocabDim : (m+1) * vocabDim]
    codebook, assigns = kmeans2(subPoints, K, iter=itsCount, minit=minit)
    return codebook, assigns

def runVQ(m, vocabDim, points, codebooks):
    subPoints = points[:, m * vocabDim : (m+1) * vocabDim]
    assigns, distortions = vq(subPoints, codebooks[m])
    return assigns, distortions

def learnCodebooksPQ(learnFilename, dim, M, K, pointsCount, codebooksFilename, \
                     threadsCount=15, itsCount=30):
    if dim % M != 0:
        raise Exception('Dim is not a multiple of M!')
    else:
        vocabDim = dim // M
    
    codebooks = np.zeros((M, K, vocabDim), dtype=np.float32)

    if isinstance(learnFilename, str):
        points = fvecs_read(learnFilename)
    else:
        points = learnFilename

    assigns = np.zeros((pointsCount, M), dtype=np.int32)

    pool = Pool(M)
    ans = pool.map(partial(runKmeans2, \
                           vocabDim=vocabDim, \
                           points=points, \
                           K = K, \
                           itsCount=itsCount, \
                           minit='++'), range(0,M))
    pool.close()
    pool.join()
    for i in range(M):
        codebooks[i,:,:] =ans[i][0]
        assigns[:,i] = ans[i][1].flatten()
    
    error = getQuantizationErrorPQ(points, assigns, codebooks)
    print(f'PQ quantation error is {error}.')
    
    with open(codebooksFilename, 'wb') as f:
        pickle.dump(codebooks, f)

    return codebooks, assigns

def encodePointsPQ(baseFilename, codebooksFilename, threadsCount=20):
    if isinstance(baseFilename, str):
        basePoints = fvecs_read(baseFilename)
    else:
        basePoints = baseFilename

    pointsCount = basePoints.shape[0]

    if isinstance(codebooksFilename, str):
        codebooks = pickle.load(open(codebooksFilename, 'rb'))
    else:
        codebooks = codebooksFilename
    
    M = codebooks.shape[0]

    vocabDim = codebooks.shape[2]

    assigns = np.zeros((pointsCount, M), dtype='int32')
    errors = np.zeros(pointsCount, dtype='float32')

    pool = Pool(M)
    ans = pool.map(partial(runVQ, \
                           vocabDim = vocabDim, \
                           points = basePoints, \
                           codebooks = codebooks), range(0,M))
    pool.close()
    pool.join()
    for i in range(M):
        assigns[:,i] = ans[i][0].flatten()
        errors += ans[i][1].flatten()**2

    return assigns, errors

def encodeDatasetPQ(baseFilename, pointsCount, codebooksFilename, codeFilename, threadsCount=20):
    codebooks = pickle.load(open(codebooksFilename, 'rb'))

    points = fvecs_read(baseFilename)
    (codes, errors) = encodePointsPQ(points, codebooks, threadsCount)
    print(f"Mean AQ quantization error: {np.mean(errors)}")
    with open(codeFilename, 'wb') as f:
        pickle.dump(codes, f)

def findNearestForRangePQ(startID, chunkSize, queriesCount, M, queryCodebookDistances, assigns, listLength):
    endID = min(startID+chunkSize, queriesCount)
    nearest = np.zeros((endID - startID, listLength), dtype='int32')
    for qid in range(startID, endID):
        distances = np.sum(queryCodebookDistances[:,qid,:][range(M), assigns], axis=1)
        nearest[qid - startID,:] = distances.argsort()[0:listLength]
    return nearest

def searchNearestNeighborsPQ(codeSource, codebookSource, querySource, queriesCount, k=1024, threadsCount=15):
    if isinstance(codebookSource, str):
        codebooks = pickle.load(open(codebookSource, 'rb'))
    else:
        codebooks = codebookSource
    
    if isinstance(codeSource, str):
        assigns = pickle.load(open(codeSource, 'rb'))
    else:
        assigns = codeSource
    
    if isinstance(querySource, str):
        queries = fvecs_read(querySource)
    else:
        queries = querySource
        
    M = codebooks.shape[0]
    K = codebooks.shape[1]
    vocabDim = codebooks.shape[2]

    queryCodebookDistances = np.zeros((M, queriesCount, K),dtype='float32')
    
    for i in range(M):
        subQueries = queries[:,i*vocabDim:(i+1)*vocabDim]
        
        q_2 = np.linalg.norm(subQueries, axis=1)**2
        q_2 = q_2.reshape(-1,1)

        q_x = subQueries @ codebooks[i].T

        x_2 = np.linalg.norm(codebooks[i], axis=1)**2
        x_2 = x_2.reshape(1,-1)

        queryCodebookDistances[i,:,:] = q_2 - 2*q_x + x_2
    
    k = min(assigns.shape[0], k)
    nearest = np.zeros((queriesCount, k), dtype='int32')
    
    if queriesCount>=threadsCount: 
        queryChunk = queriesCount // threadsCount
    else:
        queryChunk = 1

    pool = Pool(threadsCount)
    ans = pool.map(partial(findNearestForRangePQ, \
                           chunkSize = queryChunk, \
                           queriesCount = queriesCount, \
                           M = M, \
                           queryCodebookDistances = queryCodebookDistances, \
                           assigns = assigns, \
                           listLength=k), range(0, queriesCount, queryChunk))
    pool.close()
    pool.join()

    for i in range(0, queriesCount, queryChunk):
        startID = i
        endID = min(startID+queryChunk, queriesCount)
        nearest[startID:endID,:] = ans[i//queryChunk]

    return nearest

def getQuantizationErrors(points, assigns, codebooks):
    M = codebooks.shape[0]
    pointsCount = points.shape[0]
    vocabDim = codebooks.shape[2]

    errors = np.zeros(pointsCount, dtype='float32')

    pq_points = codebooks[range(M), assigns, :]
    pq_points = pq_points.reshape(pointsCount, -1)

    for i in range(M):
        tmp_error = np.linalg.norm(points[:,i*vocabDim:(i+1)*vocabDim]-pq_points[:,i*vocabDim:(i+1)*vocabDim], axis=1)**2
        errors += np.reshape(tmp_error, pointsCount)

    return errors

def getQuantizationErrorPQ(points, assigns, codebooks):
    errors = getQuantizationErrors(points, assigns, codebooks)
    return np.mean(errors)

def onlineBatch(points, startIndexPerBatch,  dim, M, K, basePointsCount, \
                  numGroup, codebooksFilename, codeFilename, \
                  threadsCount=15, itsCount=30, \
                  k=1024):
    
    if dim % M ==0:
        vocabDim = dim // M
    else:
        raise("dim % M must equals to 0.")
    
    pointsCount = points.shape[0]
    
    baseEndIndex = 0

    assigns = np.zeros((pointsCount, M), dtype='int32')
    codebooks = np.zeros((M, K, vocabDim), dtype='float32')
    codewordCount = np.zeros((M, K), dtype='int32')


    #Initialize
    initEndIndex = startIndexPerBatch[1]
    initPoints = points[:initEndIndex,:]
    print(f'init end index is {initEndIndex}.')

    initCodebooks, initAssigns = learnCodebooksPQ(initPoints, dim, M, K, \
                                                  initEndIndex, codebooksFilename, \
                                                  threadsCount, itsCount)

    codebooks[:] = initCodebooks
    assigns[:initEndIndex,:] = initAssigns
    for i in range(initAssigns.shape[0]):
        codewordCount[range(M), initAssigns[i,:]] += 1

    baseEndIndex = initEndIndex
    print(f'Finish initialize.')

    for idx,startIndex in enumerate(startIndexPerBatch):
        if idx == 0:
            continue
        print(f'group:{idx+1}/{numGroup}, new data coming.')
        #query part
        if idx < numGroup - 1:
            curBatchEndIndex = min(startIndexPerBatch[idx+1], pointsCount)
        else:
            curBatchEndIndex = pointsCount
            
        curBatchPoints = points[startIndex:curBatchEndIndex,:]
        curBatchPointsNum = curBatchPoints.shape[0]
        print(f'current batch end index is {curBatchEndIndex}.')
        print(f'current batch point number is {curBatchPointsNum}.')

        start_t = time.time()
        result = searchNearestNeighborsPQ(assigns[:baseEndIndex,:], codebooks, curBatchPoints, \
                                          curBatchPointsNum, k=1024, threadsCount=threadsCount)
        end_t = time.time()
        print(f'search nearest neighbor time: {end_t-start_t}s.')

        start_t = time.time()
        getRecall(curBatchPoints, points[:baseEndIndex,:], result)
        end_t = time.time()
        print(f'Calculate recall rate time: {end_t-start_t}s.')

        #update codebook part
        start_t = time.time()
        (curBatchAssigns, curBatchErrors) = encodePointsPQ(curBatchPoints, codebooks, threadsCount)
        end_t = time.time()
        print(f'Encode data points time: {end_t-start_t}s.')

        assigns[startIndex:curBatchEndIndex,:] = curBatchAssigns
        print(f'Current batch data encode error is {np.mean(curBatchErrors)}.')

        start_t = time.time()
        for i in range(curBatchAssigns.shape[0]):
            codewordCount[range(M), curBatchAssigns[i,:]] += 1

        for i in range(M):
            tmpAssigns = curBatchAssigns[:,i]
            for j in np.unique(tmpAssigns):
                row_index = np.where(tmpAssigns==j)[0]
                codebooks[i,j,:] = codebooks[i,j,:] + np.sum(curBatchPoints[row_index,i*vocabDim:(i+1)*vocabDim]-codebooks[i,j,:], axis=0)/codewordCount[i,j]
        end_t = time.time()
        print(f'Update codebooks time {end_t - start_t}s.')

        baseEndIndex = curBatchEndIndex

def onlineBatchSlidingWindow(points, startIndexPerBatch,  dim, M, K, basePointsCount, \
                                numGroup, codebooksFilename, codeFilename, \
                                threadsCount=15, itsCount=30, \
                                k=1000):
    
    if dim % M ==0:
        vocabDim = dim // M
    else:
        raise("dim % M must equals to 0.")
    
    pointsCount = points.shape[0]
    
    baseEndIndex = 0

    assigns = np.zeros((pointsCount, M), dtype='int32')
    codebooks = np.zeros((M, K, vocabDim), dtype='float32')
    codewordCount = np.zeros((M, K), dtype='int32')


    #Initialize
    initEndIndex = startIndexPerBatch[1]
    initPoints = points[:initEndIndex,:]
    print(f'init end index is {initEndIndex}.')

    initCodebooks, initAssigns = learnCodebooksPQ(initPoints, dim, M, K, \
                                                  initEndIndex, codebooksFilename, \
                                                  threadsCount, itsCount)

    codebooks[:] = initCodebooks
    assigns[:initEndIndex,:] = initAssigns
    for i in range(initAssigns.shape[0]):
        codewordCount[range(M), initAssigns[i,:]] += 1

    baseEndIndex = initEndIndex

    #init sliding window
    window = deque(maxlen=winSize)
    window.extend(random.sample(list(range(initEndIndex)), winSize))

    print(f'Finish initialize.')

    for idx,startIndex in enumerate(startIndexPerBatch):
        if idx == 0:
            continue
        print(f'group:{idx+1}/{numGroup}, new data coming.')
        #query part
        if idx < numGroup - 1:
            curBatchEndIndex = min(startIndexPerBatch[idx+1], pointsCount)
        else:
            curBatchEndIndex = pointsCount
            
        curBatchPoints = points[startIndex:curBatchEndIndex,:]
        curBatchPointsNum = curBatchPoints.shape[0]
        print(f'current batch end index is {curBatchEndIndex}.')
        print(f'current batch point number is {curBatchPointsNum}.')

        winIndex = np.array(window, dtype=np.int32)

        start_t = time.time()
        result = searchNearestNeighborsPQ(assigns[winIndex,:], codebooks, curBatchPoints, \
                                          curBatchPointsNum, k=k, threadsCount=threadsCount)
        end_t = time.time()
        print(f'search nearest neighbor time: {end_t-start_t}s.')

        start_t = time.time()
        getRecall(curBatchPoints, points[winIndex,:], result)
        end_t = time.time()
        print(f'Calculate recall rate time: {end_t-start_t}s.')

        #update codebook part
        start_t = time.time()
        (curBatchAssigns, curBatchErrors) = encodePointsPQ(curBatchPoints, codebooks, threadsCount)
        end_t = time.time()
        print(f'Encode data points time: {end_t-start_t}s.')

        assigns[startIndex:curBatchEndIndex,:] = curBatchAssigns
        print(f'Current batch data encode error is {np.mean(curBatchErrors)}.')

        start_t = time.time()
        for i in range(curBatchAssigns.shape[0]):
            codewordCount[range(M), curBatchAssigns[i,:]] += 1

        for i in range(M):
            tmpAssigns = curBatchAssigns[:,i]
            for j in np.unique(tmpAssigns):
                row_index = np.where(tmpAssigns==j)[0]
                codebooks[i,j,:] = codebooks[i,j,:] + np.sum(curBatchPoints[row_index,i*vocabDim:(i+1)*vocabDim]-codebooks[i,j,:], axis=0)/codewordCount[i,j]
        end_t = time.time()
        print(f'Update codebooks time {end_t - start_t}s.')

        # #update window
        oldIndex = winIndex[:curBatchPointsNum]
        window.extend(list(range(startIndex, curBatchEndIndex)))

        #update codebook based on old data
        oldBatchPoints = points[oldIndex,:]
        start_t = time.time()
        (oldBatchAssigns, oldBatchErrors) = encodePointsPQ(oldBatchPoints, codebooks, threadsCount)
        end_t = time.time()
        print(f'Encode data points time: {end_t-start_t}s.')
        print(f'Current batch data encode error is {np.mean(curBatchErrors)}.')

        start_t = time.time()
        for i in range(oldBatchAssigns.shape[0]):
            codewordCount[range(M), oldBatchAssigns[i,:]] -= 1

        for i in range(M):
            tmpAssigns = oldBatchAssigns[:,i]
            for j in np.unique(tmpAssigns):
                row_index = np.where(tmpAssigns==j)[0]
                if codewordCount[i,j] > 0:
                    codebooks[i,j,:] = codebooks[i,j,:] - np.sum(oldBatchPoints[row_index,i*vocabDim:(i+1)*vocabDim]-codebooks[i,j,:], axis=0)/codewordCount[i,j]
                elif codewordCount[i,j] < 0:
                    codewordCount[i,j] = 0
        end_t = time.time()
        print(f'Update codebooks time {end_t - start_t}s.')
        baseEndIndex = curBatchEndIndex

def onlineStreamingSlidingWindow(points, startIndexPerBatch,  dim, M, K, basePointsCount, \
                                    numGroup, codebooksFilename, codeFilename, \
                                    threadsCount=15, itsCount=30, \
                                    k=1000):
    
    if dim % M ==0:
        vocabDim = dim // M
    else:
        raise("dim % M must equals to 0.")
    
    pointsCount = points.shape[0]
    
    baseEndIndex = 0

    assigns = np.zeros((pointsCount, M), dtype='int32')
    codebooks = np.zeros((M, K, vocabDim), dtype='float32')
    codewordCount = np.zeros((M, K), dtype='int32')

    #Initialize
    initEndIndex = startIndexPerBatch[1]
    initPoints = points[:initEndIndex,:]
    print(f'init end index is {initEndIndex}.')

    initCodebooks, initAssigns = learnCodebooksPQ(initPoints, dim, M, K, \
                                                  initEndIndex, codebooksFilename, \
                                                  threadsCount, itsCount)

    codebooks[:] = initCodebooks
    assigns[:initEndIndex,:] = initAssigns
    for i in range(initAssigns.shape[0]):
        codewordCount[range(M), initAssigns[i,:]] += 1

    baseEndIndex = initEndIndex

    #init sliding window
    window = deque(maxlen=winSize)
    window.extend(random.sample(list(range(initEndIndex)), winSize))

    print(f'Finish initialize.')

    for idx,startIndex in enumerate(startIndexPerBatch):
        if idx == 0:
            continue
        print(f'group:{idx+1}/{numGroup}, new data coming.')
        #query part
        if idx < numGroup - 1:
            curBatchEndIndex = min(startIndexPerBatch[idx+1], pointsCount)
        else:
            curBatchEndIndex = pointsCount
            
        curBatchPoints = points[startIndex:curBatchEndIndex,:]
        curBatchPointsNum = curBatchPoints.shape[0]
        print(f'current batch end index is {curBatchEndIndex}.')
        print(f'current batch point number is {curBatchPointsNum}.')

        winIndex = np.array(window, dtype=np.int32)

        start_t = time.time()
        result = searchNearestNeighborsPQ(assigns[winIndex,:], codebooks, curBatchPoints, \
                                          curBatchPointsNum, k=k, threadsCount=threadsCount)
        end_t = time.time()
        print(f'search nearest neighbor time: {end_t-start_t}s.')

        start_t = time.time()
        getRecall(curBatchPoints, points[winIndex,:], result)
        end_t = time.time()
        print(f'Calculate recall rate time: {end_t-start_t}s.')

        #update codebook part
        start_t = time.time()
        (curBatchAssigns, curBatchErrors) = encodePointsPQ(curBatchPoints, codebooks, threadsCount)
        end_t = time.time()
        print(f'Encode data points time: {end_t-start_t}s.')

        assigns[startIndex:curBatchEndIndex,:] = curBatchAssigns
        print(f'Current batch data encode error is {np.mean(curBatchErrors)}.')

        start_t = time.time()
        for i in range(curBatchAssigns.shape[0]):
            codewordCount[range(M), curBatchAssigns[i,:]] += 1

        for i in range(M):
            tmpAssigns = curBatchAssigns[:,i]
            for j in np.unique(tmpAssigns):
                row_index = np.where(tmpAssigns==j)[0]
                codebooks[i,j,:] = codebooks[i,j,:] + np.sum(curBatchPoints[row_index,i*vocabDim:(i+1)*vocabDim]-codebooks[i,j,:], axis=0)/codewordCount[i,j]
        end_t = time.time()
        print(f'Update codebooks time {end_t - start_t}s.')

        # #update window
        oldIndex = winIndex[:curBatchPointsNum]
        window.extend(list(range(startIndex, curBatchEndIndex)))

        #update codebook based on old data
        oldBatchPoints = points[oldIndex,:]
        start_t = time.time()
        (oldBatchAssigns, oldBatchErrors) = encodePointsPQ(oldBatchPoints, codebooks, threadsCount)
        end_t = time.time()
        print(f'Encode data points time: {end_t-start_t}s.')
        print(f'Current batch data encode error is {np.mean(curBatchErrors)}.')

        start_t = time.time()
        for i in range(oldBatchAssigns.shape[0]):
            codewordCount[range(M), oldBatchAssigns[i,:]] -= 1

        for i in range(M):
            tmpAssigns = oldBatchAssigns[:,i]
            for j in np.unique(tmpAssigns):
                row_index = np.where(tmpAssigns==j)[0]
                if codewordCount[i,j] > 0:
                    codebooks[i,j,:] = codebooks[i,j,:] - np.sum(oldBatchPoints[row_index,i*vocabDim:(i+1)*vocabDim]-codebooks[i,j,:], axis=0)/codewordCount[i,j]
                elif codewordCount[i,j] < 0:
                    codewordCount[i,j] = 0
        end_t = time.time()
        print(f'Update codebooks time {end_t - start_t}s.')
        baseEndIndex = curBatchEndIndex


def onlineBatchNoUpdate(points, startIndexPerBatch,  dim, M, K, basePointsCount, \
                        numGroup, codebooksFilename, codeFilename, \
                        threadsCount=15, itsCount=30, \
                        k=1024):
    if dim % M ==0:
        vocabDim = dim // M
    else:
        raise("dim % M must equals to 0.")
    
    pointsCount = points.shape[0]
    
    baseEndIndex = 0

    assigns = np.zeros((pointsCount, M), dtype='int32')
    codebooks = np.zeros((M, K, vocabDim), dtype='float32')
    codewordCount = np.zeros((M, K), dtype='int32')


    #Initialize
    initEndIndex = startIndexPerBatch[1]
    initPoints = points[:initEndIndex,:]
    print(f'init end index is {initEndIndex}.')

    initCodebooks, initAssigns = learnCodebooksPQ(initPoints, dim, M, K, \
                                                  initEndIndex, codebooksFilename, \
                                                  threadsCount, itsCount)

    codebooks[:] = initCodebooks
    assigns[:initEndIndex,:] = initAssigns
    for i in range(initAssigns.shape[0]):
        codewordCount[range(M), initAssigns[i,:]] += 1

    baseEndIndex = initEndIndex
    print(f'Finish initialize.')

    for idx,startIndex in enumerate(startIndexPerBatch):
        if idx == 0:
            continue
        print(f'group:{idx+1}/{numGroup}, new data coming.')
        #query part
        if idx < numGroup - 1:
            curBatchEndIndex = min(startIndexPerBatch[idx+1], pointsCount)
        else:
            curBatchEndIndex = pointsCount
            
        curBatchPoints = points[startIndex:curBatchEndIndex,:]
        curBatchPointsNum = curBatchPoints.shape[0]
        print(f'current batch end index is {curBatchEndIndex}.')
        print(f'current batch point number is {curBatchPointsNum}.')

        start_t = time.time()
        result = searchNearestNeighborsPQ(assigns[:baseEndIndex,:], codebooks, curBatchPoints, \
                                          curBatchPointsNum, k=1024, threadsCount=threadsCount)
        end_t = time.time()
        print(f'search nearest neighbor time: {end_t-start_t}s.')

        start_t = time.time()
        getRecall(curBatchPoints, points[:baseEndIndex,:], result)
        end_t = time.time()
        print(f'Calculate recall rate time: {end_t-start_t}s.')

        #update codebook part
        start_t = time.time()
        (curBatchAssigns, curBatchErrors) = encodePointsPQ(curBatchPoints, codebooks, threadsCount)
        end_t = time.time()
        print(f'Encode data points time: {end_t-start_t}s.')

        assigns[startIndex:curBatchEndIndex,:] = curBatchAssigns
        print(f'Current batch data encode error is {np.mean(curBatchErrors)}.')

        # start_t = time.time()
        # for i in range(curBatchAssigns.shape[0]):
        #     codewordCount[range(M), curBatchAssigns[i,:]] += 1

        # for i in range(M):
        #     tmpAssigns = curBatchAssigns[:,i]
        #     for j in np.unique(tmpAssigns):
        #         row_index = np.where(tmpAssigns==j)[0]
        #         codebooks[i,j,:] = codebooks[i,j,:] + np.sum(curBatchPoints[row_index,i*vocabDim:(i+1)*vocabDim]-codebooks[i,j,:], axis=0)/codewordCount[i,j]
        # end_t = time.time()
        # print(f'Update codebooks time {end_t - start_t}s.')

        baseEndIndex = curBatchEndIndex

def onlineBatchPQ(baseFilename, dim, M, K, basePointsCount, \
                  numGroup, codebooksFilename, codeFilename, \
                  threadsCount=15, itsCount=30, \
                  k=1024, MODE='update'):
    if M < 1:
        raise Exception("M is not positive.")
    
    #sift1m dataset
    if baseFilename.endswith('.fvecs'):
        points = fvecs_read(baseFilename)
        pointsCount = points.shape[0]
        numPerGroup = pointsCount // numGroup
        startIndexPerBatch = np.array([numPerGroup*i for i in range(numGroup)], dtype='int32')

    #newsgroup20 dataset
    elif baseFilename.endswith('newsgroup20.npy'):
        points = np.load(baseFilename)
        pointsCount = points.shape[0]
        
        numPerGroup = pointsCount // numGroup
        startIndexPerBatch = np.array([numPerGroup*i for i in range(numGroup)], dtype='int32')

    #caltech101 dataset  
    elif baseFilename.endswith('caltech101.mat'):
        data = matIO.loadmat(baseFilename)
        rawdata = data['gist']
        points = np.zeros((basePointsCount, dim))
        if 'win' in MODE:
            k = 1000
            startIndexPerBatch, points = genStartIndexPerBatchSW(rawdata, points, numGroup, classNum=102, is_shuffle=True)
        else:
            startIndexPerBatch, points = genStartIndexPerBatch(rawdata, points, numGroup, classNum=102, is_shuffle=True)
        # print(startIndexPerBatch)
        # for i in range(1, numGroup):
        #     print(startIndexPerBatch[i] - startIndexPerBatch[i-1])
        # return

    #halfdome dataset
    elif baseFilename.endswith('halfdome.mat'):
        data = matIO.loadmat(baseFilename)
        rawDataWithNan = data['data']
        df = pd.DataFrame(rawDataWithNan)
        newdf = df.dropna(axis=0, how='any')
        rawdata = newdf.values
        
        assert(rawdata.shape[0] == basePointsCount)

        points = np.zeros((basePointsCount, dim))

        if 'win' in MODE:
            k = 10000
            startIndexPerBatch, points = genStartIndexPerBatchHalfDomeSW(rawdata,'./data/halfdome/info.txt', points, numGroup, classNum=28086, is_shuffle=True)
        else:
            startIndexPerBatch, points = genStartIndexPerBatchHalfDome(rawdata,'./data/halfdome/info.txt', points, numGroup, classNum=28086, is_shuffle=True)
        # print(startIndexPerBatch)
        # for i in range(1, numGroup):
        #     print(startIndexPerBatch[i] - startIndexPerBatch[i-1])
        # return

    elif baseFilename.endswith('sun397.mat'):
        data = matIO.loadmat(baseFilename)
        rawdata = data['gist']
        points = np.zeros((basePointsCount, dim))
        if 'win' in MODE:
            k = 10000
            startIndexPerBatch, points = genStartIndexPerBatchSun397SW(rawdata, points, numGroup, classNum=397, is_shuffle=True)
        else:
            startIndexPerBatch, points = genStartIndexPerBatchSun397(rawdata, points, numGroup, classNum=397, is_shuffle=True)
        # print(startIndexPerBatch)
        # for i in range(1, numGroup-1):
        #     print(startIndexPerBatch[i] - startIndexPerBatch[i-1])
        # print(basePointsCount - startIndexPerBatch[numGroup-1])
        # return
    
    elif baseFilename.endswith('cifar10.mat'):
        data = matIO.loadmat(baseFilename)
        rawdata = data['gist']
        points = np.zeros((basePointsCount, dim))
        if 'win' in MODE:
            k = 10000
            startIndexPerBatch, points = genStartIndexPerBatchCifar10SW(rawdata, points, numGroup, classNum=10, is_shuffle=True)
        else:
            startIndexPerBatch, points = genStartIndexPerBatchCifar10(rawdata, points, numGroup, classNum=10, is_shuffle=True)
        # print(startIndexPerBatch)
        # for i in range(1, numGroup-1):
        #     print(startIndexPerBatch[i] - startIndexPerBatch[i-1])
        # print(basePointsCount - startIndexPerBatch[numGroup-1])
        # return
    
    elif baseFilename.endswith('imagenet.npy'):
        rawdata = np.load(baseFilename, allow_pickle=True)
        points = np.zeros((basePointsCount, dim))
        startIndexPerBatch, points = genStartIndexPerBatchImageNet(rawdata, points, numGroup, classNum=1000, is_shuffle=True)

    else:
        raise('Read data error.')

    print(f"read from {baseFilename}, points's shape is {points.shape}.")

    if MODE == 'update':
        onlineBatch(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                    codebooksFilename, codeFilename, \
                    threadsCount=threadsCount, itsCount=itsCount, \
                    k=k)

    elif MODE == 'noupdate':
        onlineBatchNoUpdate(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                            codebooksFilename, codeFilename, \
                            threadsCount=threadsCount, itsCount=itsCount, \
                            k=k)
    
    elif MODE == 'win_batch':
        onlineBatchSlidingWindow(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                            codebooksFilename, codeFilename, \
                            threadsCount=threadsCount, itsCount=itsCount, \
                            k=k)

    else:
        raise(f"Unexpected MODE {MODE}.")
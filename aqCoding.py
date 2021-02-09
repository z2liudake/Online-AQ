import time
import pickle
import random
from functools import partial
from itertools import combinations
from multiprocessing import Pool

from collections import deque

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy import io as matIO
from scipy.sparse.linalg import lsmr
from scipy.stats import ortho_group
from scipy.linalg import pinv

from utils import *
from newIO import *
from getRecallOnline import *
from Pretrain.model import *

maxIter = 1
numCB = 5

group=2
"""
1. 2: 4 codebooks per group
2. 4: 2 codebooks per group
"""

encodeMode = 'beam_search'
"""
1. hill climb: hill_climb
2. beam search: beam_search
3. hill climb + beam search: beam_hill
4. simple beam: beam_simple
5. simple beam + sample beam search: beam_sample
"""


def solveDimensionLeastSquares(startDim, dimCount, data, indices, indptr, trainPoints, codebookSize, M):
    A = sparse.csr_matrix((data, indices, indptr), shape=(trainPoints.shape[0], M*codebookSize), copy=False)
    discrepancy = 0
    dimCount = min(dimCount, trainPoints.shape[1] - startDim)
    codebooksComponents = np.zeros((M, codebookSize, dimCount), dtype='float32')
    for dim in range(startDim, startDim+dimCount):
        b = trainPoints[:, dim].flatten()
        solution = lsmr(A, b, show=False, maxiter=250)
        codebooksComponents[:, :, dim-startDim] = np.reshape(solution[0], (M, codebookSize))
        discrepancy += solution[3] ** 2
    return (codebooksComponents, discrepancy)

def getMeanQuantizationError(points, assigns, codebooks):
    errors = getQuantizationErrors(points, assigns, codebooks)
    return np.mean(errors)

def getQuantizationErrors(points, assigns, codebooks):
    pointsCopy = points.copy()
    for m in range(codebooks.shape[0]):
        pointsCopy = pointsCopy - codebooks[m,assigns[:,m],:]
    errors = np.zeros((points.shape[0]), dtype='float32')
    for pid in range(points.shape[0]):
        errors[pid] = np.dot(pointsCopy[pid,:], pointsCopy[pid,:].T)
    return errors

def encodePointHillClimbing(startPid, pointsCount, pointCodebookProducts, codebooksProducts, codebooksNorms):
    M = codebooksProducts.shape[0]
    K = codebooksProducts.shape[1]

    pointsCount = min(pointsCount, pointCodebookProducts.shape[0] - startPid)

    assigns = np.zeros((pointsCount, M), dtype='int32')
    errors = np.zeros(pointsCount, dtype='float32')
    
    M_index = np.arange(0,M*K,K, dtype=np.int32)

    for pid in range(startPid, startPid + pointsCount):
        curAssigns = np.random.randint(low=0, high=K, size=M, dtype=np.int32)
        curError =  - pointCodebookProducts[pid, M_index + curAssigns].sum() + codebooksNorms[M_index + curAssigns].sum()
        for i in range(M-1):
            curError += codebooksProducts[i,curAssigns[i],curAssigns[i+1:]+M_index[i+1:]].sum()

        bestAssigns = curAssigns.copy()
        
        for _ in range(5):
            for idx in range(M):
                tmpError = curError + pointCodebookProducts[pid, curAssigns[idx]+M_index[idx]]
                tmpError -= codebooksProducts[idx, curAssigns[idx], curAssigns+M_index].sum()
                tmpError += codebooksNorms[curAssigns[idx]+M_index[idx]] 
                for j in range(K):
                    curAssigns[idx] = j
                    incError =  -pointCodebookProducts[pid, j+M_index[idx]] + codebooksProducts[idx, j, curAssigns+M_index].sum() \
                                - codebooksNorms[j + M_index[idx]]
                    tmpError += incError

                    if tmpError < curError:
                        curError = tmpError
                        bestAssigns[idx] = j
                    
                    tmpError -= incError

                curAssigns[idx] = bestAssigns[idx]

        assigns[pid-startPid,:] = bestAssigns
        errors[pid-startPid] = curError
    
    return (assigns, errors)

def encodePointsSimpleBeamSearch(startPid, pointsCount, pointCodebookProducts, codebooksProducts, codebooksNorms, branch):
    """
        Simple Beam Search
        startPid: 当前需要编码的一组向量的起始index
        pointsCount: 当前需要编码的向量个数
        pointCodebookProducts: 编码向量与codebook里每个codeword的内积
        codebooksProducts：codebook里codeword之间的内积的2倍
        codebooksNorms：codebook里每个codeword与自身的内积
        branch：Beam Search中的参数 N
    """
    M = codebooksProducts.shape[0]
    K = codebooksProducts.shape[1]

    pointsCount = min(pointsCount, pointCodebookProducts.shape[0] - startPid)

    assigns = np.zeros((pointsCount, M), dtype='int32')
    errors = np.zeros((pointsCount), dtype='float32')

    for pid in range(startPid, startPid+pointsCount):
        #初始化计算x与第一个码书中的每个码字的距离
        distances = - pointCodebookProducts[pid,0:K] + codebooksNorms[0:K]

        # 初始的与待编码向量最接近的branch个codeword
        bestIdx = distances.argsort()[0:branch]

        #记录codeword index的 branch 个tuple
        bestSums = -1 * np.ones((branch, M), dtype='int32')
        bestSums[:,0] = bestIdx
        
        #记录branch 个 tuple 当前与x之间的距离
        bestSumScores = distances[bestIdx]
        
        for m in range(1, M):
            tmp_score = bestSumScores.reshape(-1,1)  \
                        - pointCodebookProducts[pid,m*K:(m+1)*K].reshape(1,-1)  \
                        + codebooksNorms[m*K:(m+1)*K].reshape(1,-1)
            for t in range(m):
                tmp_score += codebooksProducts[t, bestSums[:,t].reshape(-1,1), np.arange(m*K,(m+1)*K).reshape(1,-1)]

            tmp_bestIdx = np.argsort(tmp_score.flatten())[0:branch]

            newBestSums = -1 * np.ones((branch, M), dtype='int32')

            newBestSumsScores = tmp_score.flatten()[tmp_bestIdx]

            newBestSums = bestSums[tmp_bestIdx//K, :]

            newBestSums[:,m] = tmp_bestIdx % K

            bestSums = newBestSums.copy()
            bestSumScores = newBestSumsScores.copy()

        assigns[pid-startPid,:] = bestSums[0,:]
        errors[pid-startPid] = bestSumScores[0]

    return (assigns, errors)

def encodePointsBeamSearch(startPid, pointsCount, pointCodebookProducts, codebooksProducts, codebooksNorms, branch):
    """
        startPid: 当前需要编码的一组向量的起始index
        pointsCount: 当前需要编码的向量个数
        pointCodebookProducts: 编码向量与codebook里每个codeword的内积
        codebooksProducts：codebook里codeword之间的内积的2倍
        codebooksNorms：codebook里每个codeword与自身的内积
        branch：Beam Search中的参数 N
    """
    M = codebooksProducts.shape[0]
    K = codebooksProducts.shape[1]

    hashArray = np.array([13 ** i for i in range(M)])
    pointsCount = min(pointsCount, pointCodebookProducts.shape[0] - startPid)

    assigns = np.zeros((pointsCount, M), dtype='int32')
    errors = np.zeros((pointsCount), dtype='float32')

    for pid in range(startPid, startPid+pointsCount):

        distances = - pointCodebookProducts[pid,:] + codebooksNorms


        bestIdx = distances.argsort()[0:branch]


        vocIds = bestIdx // K

        wordIds = bestIdx % K

        bestSums = -1 * np.ones((branch, M), dtype='int32')
        for candidateIdx in range(branch):
            bestSums[candidateIdx,vocIds[candidateIdx]] = wordIds[candidateIdx]
        

        bestSumScores = distances[bestIdx]
        

        for _ in range(1, M):

            candidatesScores = np.array([bestSumScores[i].repeat(M * K) for i in range(branch)]).flatten()
            candidatesScores += np.tile(distances, branch)


            globalHashTable = np.zeros(115249, dtype='int8')


            for candidateIdx in range(branch):
                for m in range(M):
                      if bestSums[candidateIdx,m] < 0:
                          continue
                      candidatesScores[candidateIdx*M*K:(candidateIdx+1)*M*K] += \
                          codebooksProducts[m, bestSums[candidateIdx,m], :]
                     
                      candidatesScores[candidateIdx*M*K + m*K:candidateIdx*M*K+(m+1)*K] += 999999
            bestIndices = candidatesScores.argsort()
            found = 0
            currentBestIndex = 0
            newBestSums = -1 * np.ones((branch, M), dtype='int32')
            newBestSumsScores = -1 * np.ones((branch), dtype='float32')
            while found < branch:
                bestIndex = bestIndices[currentBestIndex]
                candidateId = bestIndex // (M * K)
                codebookId = (bestIndex % (M * K)) // K
                wordId = (bestIndex % (M * K)) % K
                bestSums[candidateId,codebookId] = wordId
                hashIdx = np.dot(bestSums[candidateId,:], hashArray) % 115249

                if globalHashTable[hashIdx] == 1:
                    bestSums[candidateId,codebookId] = -1
                    currentBestIndex += 1
                    continue
                else:
                    bestSums[candidateId,codebookId] = -1
                    globalHashTable[hashIdx] = 1
                    newBestSums[found,:] = bestSums[candidateId,:]
                    newBestSums[found,codebookId] = wordId
                    newBestSumsScores[found] = candidatesScores[bestIndex]
                    found += 1
                    currentBestIndex += 1
            bestSums = newBestSums.copy()
            bestSumScores = newBestSumsScores.copy()
        assigns[pid-startPid,:] = bestSums[0,:]
        errors[pid-startPid] = bestSumScores[0]
    return (assigns, errors)

def encodePointsHillBeamSearch(startPid, pointsCount, pointCodebookProducts, codebooksProducts, codebooksNorms, branch, group):
    M = codebooksProducts.shape[0]
    K = codebooksProducts.shape[1]

    pointsCount = min(pointsCount, pointCodebookProducts.shape[0] - startPid)

    assigns = np.zeros((pointsCount, M), dtype='int32')
    errors = np.zeros(pointsCount, dtype='float32')

    M_index = np.arange(0,M*K,K, dtype=np.int32)
    hashArray = np.array([13 ** i for i in range(M)])

    for pid in range(startPid, startPid + pointsCount):
        #initialize with hill climbe
        curAssigns = np.random.randint(low=0, high=K, size=M, dtype=np.int32)
        curError =  - pointCodebookProducts[pid, M_index + curAssigns].sum() + codebooksNorms[M_index + curAssigns].sum()
        for i in range(M-1):
            curError += codebooksProducts[i,curAssigns[i],curAssigns[i+1:]+M_index[i+1:]].sum()

        bestAssigns = curAssigns.copy()
        
        for idx in range(M):
            tmpError = curError + pointCodebookProducts[pid, curAssigns[idx]+M_index[idx]]
            tmpError -= codebooksProducts[idx, curAssigns[idx], curAssigns+M_index].sum()
            tmpError += codebooksNorms[curAssigns[idx]+M_index[idx]]
            for j in range(K):
                curAssigns[idx] = j
                incError =  -pointCodebookProducts[pid, j + M_index[idx]] + codebooksProducts[idx, j, curAssigns+M_index].sum() \
                            - codebooksNorms[j + M_index[idx]]
                tmpError += incError

                if tmpError < curError:
                    curError = tmpError
                    bestAssigns[idx] = j
                
                tmpError -= incError

            curAssigns[idx] = bestAssigns[idx]
        
        #Block Beam Search
        cbPerGroup = M // group
        C_index = list(range(M))
        random.shuffle(C_index)
        C_index = np.array(C_index, dtype=np.int32)

        distances = - pointCodebookProducts[pid,:] + codebooksNorms

        for g in range(group):
            localIndex = C_index[g * cbPerGroup: (g+1) * cbPerGroup]
            bestSums = -1 * np.ones((branch, M), dtype=np.int32)
            bestSums[:] = bestAssigns
            bestSums[:,localIndex] = -1
            #Accurate error
            for idx1,t1 in enumerate(bestAssigns):
                if t1 in localIndex:
                    continue
                fixedError = -pointCodebookProducts[pid, t1+idx1*K] + codebooksNorms[t1+idx1*K]

                for idx2,t2 in enumerate(bestAssigns):
                    if idx2 <= idx1 or t2 in localIndex:
                        continue
                    fixedError += codebooksProducts[idx1, t1, idx2*K+t2]

            bestSumscores = np.ones(branch) * fixedError

            helpIndex = np.hstack([np.arange(i*K, (i+1)*K, dtype=np.int32) for i in localIndex])

            for j in range(cbPerGroup):
                candidatesScores = np.array([bestSumscores[i].repeat(cbPerGroup*K) for i in range(branch)]).flatten()
                candidatesScores += np.tile(distances[helpIndex], branch)

                globalHashTable = np.zeros(115249, dtype='int8')
                
                #check for right
                for candidateIdx in range(branch):
                    for m in range(M):
                        if bestSums[candidateIdx, m] < 0:
                            continue
                        candidatesScores[candidateIdx*cbPerGroup*K:(candidateIdx+1)*cbPerGroup*K] += \
                            codebooksProducts[m, bestSums[candidateIdx,m], helpIndex]
                        
                        if m in localIndex:
                            position = np.where(localIndex == m)[0]
                            position = position.item()
                            candidatesScores[candidateIdx*cbPerGroup*K+position*K:candidateIdx*cbPerGroup*K+(position+1)*K] += 999999
                if j == 0:
                    bestIndices = candidatesScores[0:cbPerGroup*K].argsort()
                else:
                    bestIndices = candidatesScores.argsort()
                found = 0
                currentBestIndex = 0
                newBestSums = -1 * np.ones((branch, M), dtype='int32')
                newBestSumsScores = -1 * np.ones((branch), dtype='float32')

                while found < branch:
                    bestIndex = bestIndices[currentBestIndex]
                    candidateId = bestIndex // (cbPerGroup * K)
                    codebookId = (bestIndex % (cbPerGroup * K)) // K
                    wordId = (bestIndex % (cbPerGroup * K)) % K
                    bestSums[candidateId,localIndex[codebookId]] = wordId
                    hashIdx = np.dot(bestSums[candidateId,:], hashArray) % 115249

                    if globalHashTable[hashIdx] == 1:
                        bestSums[candidateId,localIndex[codebookId]] = -1
                        currentBestIndex += 1
                        continue
                    else:
                        bestSums[candidateId,localIndex[codebookId]] = -1
                        globalHashTable[hashIdx] = 1
                        newBestSums[found,:] = bestSums[candidateId,:]
                        newBestSums[found,localIndex[codebookId]] = wordId
                        newBestSumsScores[found] = candidatesScores[bestIndex]
                        found += 1
                        currentBestIndex += 1
                bestSums = newBestSums.copy()
                bestSumScores = newBestSumsScores.copy()

            bestAssigns[:] = bestSums[0]
            curError = bestSumScores[0]

        assigns[pid-startPid,:] = bestAssigns
        errors[pid-startPid] = curError
    return (assigns, errors)

def encodePointsSampleBeamSearch(startPid, pointsCount, pointCodebookProducts, codebooksProducts, codebooksNorms, branch):
    M = codebooksProducts.shape[0]
    K = codebooksProducts.shape[1]

    hashArray = np.array([13 ** i for i in range(M)])
    pointsCount = min(pointsCount, pointCodebookProducts.shape[0] - startPid)

    assigns = np.zeros((pointsCount, M), dtype='int32')
    errors = np.zeros((pointsCount), dtype='float32')

    for pid in range(startPid, startPid+pointsCount):
        bestAssigns,curError = metaSimpleBeamSearch(pointCodebookProducts[pid], codebooksProducts, codebooksNorms, branch)

        distances = - pointCodebookProducts[pid,:] + codebooksNorms

        for _ in range(maxIter):
            #selectCB: codebooks to be beam search
            selectCB = np.sort(np.array(random.sample(list(range(M)), numCB), dtype=np.int32))
            helpIndex = np.hstack([np.arange(i*K,(i+1)*K, dtype=np.int32) for i in selectCB])

            bestSums = -1 * np.ones((branch, M), dtype=np.int32)
            bestSums[:] = bestAssigns
            bestSums[:,selectCB] = -1

            #error on fixed index
            for idx1,t1 in enumerate(bestAssigns):
                if t1 in selectCB:
                    continue
                fixedError = -pointCodebookProducts[pid, t1+idx1*K] + codebooksNorms[t1+idx1*K]

                for idx2,t2 in enumerate(bestAssigns):
                    if idx2 <= idx1 or t2 in selectCB:
                        continue
                    fixedError += codebooksProducts[idx1, t1, idx2*K+t2]

            bestSumscores = np.ones(branch) * fixedError

            for j in range(numCB):
                candidatesScores = np.array([bestSumscores[i].repeat(numCB*K) for i in range(branch)]).flatten()
                candidatesScores += np.tile(distances[helpIndex], branch)

                globalHashTable = np.zeros(115249, dtype='int8')

                #check for right
                for candidateIdx in range(branch):
                    for m in range(M):
                        if bestSums[candidateIdx, m] < 0:
                            continue
                        candidatesScores[candidateIdx*(numCB*K):(candidateIdx+1)*(numCB*K)] += \
                            codebooksProducts[m, bestSums[candidateIdx, m], helpIndex]
                        
                        if m in selectCB:
                            position = np.where(selectCB == m)[0]
                            position = position.item()
                            candidatesScores[candidateIdx*numCB*K+position*K:candidateIdx*numCB*K+(position+1)*K] += 999999

                if j == 0:
                    bestIndices = candidatesScores[0:numCB*K].argsort()
                else:
                    bestIndices = candidatesScores.argsort()
                
                found = 0
                currentBestIndex = 0
                newBestSums = -1 * np.ones((branch, M), dtype='int32')
                newBestSumsScores = -1 * np.ones((branch), dtype='float32')

                while found < branch:
                    bestIndex = bestIndices[currentBestIndex]
                    candidateId = bestIndex // (numCB * K)
                    codebookId = (bestIndex % (numCB * K)) // K
                    wordId = (bestIndex % (numCB * K)) % K
                    bestSums[candidateId,selectCB[codebookId]] = wordId
                    hashIdx = np.dot(bestSums[candidateId,:], hashArray) % 115249

                    if globalHashTable[hashIdx] == 1:
                        bestSums[candidateId,selectCB[codebookId]] = -1
                        currentBestIndex += 1
                        continue
                    else:
                        bestSums[candidateId,selectCB[codebookId]] = -1
                        globalHashTable[hashIdx] = 1
                        newBestSums[found,:] = bestSums[candidateId,:]
                        newBestSums[found,selectCB[codebookId]] = wordId
                        newBestSumsScores[found] = candidatesScores[bestIndex]
                        found += 1
                        currentBestIndex += 1

                bestSums = newBestSums.copy()
                bestSumScores = newBestSumsScores.copy()
        
            if bestSumScores[0] < curError:
                bestAssigns[:] = bestSums[0]
                curError = bestSumScores[0]

        assigns[pid-startPid,:] = bestAssigns
        errors[pid-startPid] = curError
    
    return (assigns, errors)

def metaSimpleBeamSearch(pointCodebooksProduct, codebooksProducts, codebooksNorms, branch):
    M = codebooksProducts.shape[0]
    K = codebooksProducts.shape[1]

    #初始化计算x与第一个码书中的每个字的距离
    distances = - pointCodebooksProduct[0:K] + codebooksNorms[0:K]
    bestIdx = distances.argsort()[0:branch]

    #记录codeword index的 branch 个tuple
    bestSums = -1 * np.ones((branch, M), dtype='int32')
    bestSums[:,0] = bestIdx

    #记录branch 个 tuple 当前与x之间的距离
    bestSumScores = distances[bestIdx] 

    for m in range(1, M):
        tmp_score = bestSumScores.reshape(-1,1)  \
                        - pointCodebooksProduct[m*K:(m+1)*K].reshape(1,-1)  \
                        + codebooksNorms[m*K:(m+1)*K].reshape(1,-1)
        
        for t in range(m):
                tmp_score += codebooksProducts[t, bestSums[:,t].reshape(-1,1), np.arange(m*K,(m+1)*K).reshape(1,-1)]

        tmp_bestIdx = np.argsort(tmp_score.flatten())[0:branch]

        newBestSums = -1 * np.ones((branch, M), dtype='int32')

        newBestSumsScores = tmp_score.flatten()[tmp_bestIdx]

        newBestSums = bestSums[tmp_bestIdx//K, :]

        newBestSums[:,m] = tmp_bestIdx % K

        bestSums = newBestSums.copy()
        bestSumScores = newBestSumsScores.copy()

    return bestSums[0], bestSumScores[0]

def encodePointsAQ(points, codebooks, branch, mode=encodeMode):
    """
        mode: 
            1. hill climb: hill_climb
            2. beam search: beam_search
            3. hill climb + beam search: beam_hill
            4. simple beam: beam_simple
    """
    pointsCount = points.shape[0]
    M = codebooks.shape[0]
    K = codebooks.shape[1]

    # 每个codeword与其他所有codeword的内积的2倍
    codebooksProducts = np.zeros((M,K,M*K), dtype='float32')
    # 中间的辅助变量，用以记录每个codeword与其他所有codeword的内积的2倍
    fullProducts = np.zeros((M,K,M,K), dtype='float32')
    # 每个codeword与自身的内积
    codebooksNorms = np.zeros((M*K), dtype='float32')

    for m1 in range(M):
        for m2 in range(M):
            fullProducts[m1,:,m2,:] = 2 * np.dot(codebooks[m1,:,:], codebooks[m2,:,:].T)
        codebooksNorms[m1*K:(m1+1)*K] = fullProducts[m1,:,m1,:].diagonal() / 2
        codebooksProducts[m1,:,:] = np.reshape(fullProducts[m1,:,:,:], (K,M*K))

    #用以记录新的assigns
    assigns = np.zeros((pointsCount, M), dtype='int32')

    #分批次对points进行encode
    pidChunkSize = min(pointsCount, 5030)

    #用以记录重新编码后量化误差的向量
    errors = np.zeros(pointsCount, dtype='float32')

    for startPid in range(0, pointsCount, pidChunkSize):
        
        #当前循环中有多少个points需要编码
        realChunkSize = min(pidChunkSize, pointsCount - startPid)
        
        #当前循环中需要编码的points
        chunkPoints = points[startPid:startPid+realChunkSize,:]

        #需要编码的points和每个codeword的内积
        queryProducts = np.zeros((realChunkSize, M * K), dtype=np.float32)
        for pid in range(realChunkSize):
            errors[pid+startPid] += np.dot(chunkPoints[pid,:], chunkPoints[pid,:].T)
        for m in range(M):
            queryProducts[:,m*K:(m+1)*K] = 2 * np.dot(chunkPoints, codebooks[m,:,:].T)

        #进程数量
        poolSize = 8
        if realChunkSize>=poolSize:
            chunkSize = realChunkSize // poolSize
        else:
            chunkSize = 1
        pool = Pool(processes=poolSize+1)
        if mode == 'beam_search':
            ans = pool.map_async(partial(encodePointsBeamSearch, \
                                pointsCount=chunkSize, \
                                pointCodebookProducts=queryProducts, \
                                codebooksProducts=codebooksProducts, \
                                codebooksNorms=codebooksNorms, \
                                branch=branch), range(0, realChunkSize, chunkSize)).get()
        
        elif mode == 'hill_climb':
            ans = pool.map_async(partial(encodePointHillClimbing, \
                                pointsCount=chunkSize, \
                                pointCodebookProducts=queryProducts, \
                                codebooksProducts=codebooksProducts, \
                                codebooksNorms=codebooksNorms), range(0, realChunkSize, chunkSize)).get()
        
        elif mode == 'beam_hill':
            ans = pool.map_async(partial(encodePointsHillBeamSearch, \
                                pointsCount=chunkSize, \
                                pointCodebookProducts=queryProducts, \
                                codebooksProducts=codebooksProducts, \
                                codebooksNorms=codebooksNorms, \
                                branch=branch, \
                                group=group), range(0, realChunkSize, chunkSize)).get()

        elif mode == 'beam_simple':
            ans = pool.map_async(partial(encodePointsSimpleBeamSearch, \
                                pointsCount=chunkSize, \
                                pointCodebookProducts=queryProducts, \
                                codebooksProducts=codebooksProducts, \
                                codebooksNorms=codebooksNorms, \
                                branch=branch), range(0, realChunkSize, chunkSize)).get()
        elif mode == 'beam_sample':
            ans = pool.map_async(partial(encodePointsSampleBeamSearch, \
                                pointsCount=chunkSize, \
                                pointCodebookProducts=queryProducts, \
                                codebooksProducts=codebooksProducts, \
                                codebooksNorms=codebooksNorms, \
                                branch=branch), range(0, realChunkSize, chunkSize)).get()

        else:
            raise Exception(f"Unexcepted mode:{mode}.")

        pool.close()
        pool.join()
        for startChunkPid in range(0, realChunkSize, chunkSize):
            pidsCount = min(chunkSize, realChunkSize - startChunkPid)
            assigns[startPid+startChunkPid:startPid+startChunkPid+pidsCount,:] = ans[startChunkPid//chunkSize][0]
            errors[startPid+startChunkPid:startPid+startChunkPid+pidsCount] += ans[startChunkPid//chunkSize][1]
    
    return (assigns, errors)

def learnCodebooksAQ(points, dim, M, K, pointsCount, codebooksFilename, branch, saveCodebook=True, threadsCount=15, itsCount=20,CODEBOOKS=None,ASSIGNS=None):
    if M < 1:
        raise Exception('M is not positive!')
    
    assigns = np.zeros((pointsCount, M), dtype='int32')
    codebooks = np.zeros((M, K, dim), dtype='float32')
    if CODEBOOKS is not None:
        codebooks = CODEBOOKS.copy()
    # random initialization of assignment variables
    # (initializations from (O)PQ should be used for better results)
    for m in range(M):
        assigns[:,m] = np.random.randint(0, K, pointsCount)

    if ASSIGNS is not None:
        prePointNum = ASSIGNS.shape[0]
        assigns[:prePointNum] = ASSIGNS.copy()

    errors = getQuantizationErrors(points, assigns, codebooks)
    print (f"Error before learning iterations: {np.mean(errors)}")
    
    data = np.ones(M * pointsCount, dtype='float32')
    indices = np.zeros(M * pointsCount, dtype='int32')
    indptr = np.array(range(0, pointsCount + 1)) * M
    
    oldAssigns = assigns.copy()
    
    for it in range(itsCount):
        for i in range(pointsCount * M):
            indices[i] = 0
        for pid in range(pointsCount):
            for m in range(M):
                indices[pid * M + m] = m * K + assigns[pid,m]

        dimChunkSize = dim // threadsCount
        pool = Pool(threadsCount)
        ans = pool.map(partial(solveDimensionLeastSquares, \
                               dimCount=dimChunkSize, \
                               data=data, \
                               indices=indices, \
                               indptr=indptr, \
                               trainPoints=points, \
                               codebookSize=K, M=M), range(0, dim, dimChunkSize))
        pool.close()
        pool.join()
        for d in range(0, dim, dimChunkSize):
            dimCount = min(dimChunkSize, dim - d)
            codebooks[:, :, d:d+dimCount] = ans[d // dimChunkSize][0]
        
        oldAssigns = assigns.copy()
        
        errors = getQuantizationErrors(points, assigns, codebooks)
        print(f"Iterations:{it}/{itsCount},Error after LSMR step: {np.mean(errors)}")
        (assigns, errors) = encodePointsAQ(points, codebooks, branch, mode=encodeMode)
        errors = getQuantizationErrors(points, assigns, codebooks)
        print(f"Iterations:{it}/{itsCount},Error after encoding step: {np.mean(errors)}")
        
    if saveCodebook:
        with open(codebooksFilename, 'wb') as f:
            pickle.dump(codebooks, f)
    
        codesFilename = codebooksFilename.replace('codebooks','codes')
        with open(codesFilename, 'wb') as f:
            pickle.dump(assigns, f)
        
        oldCodesFilename = codebooksFilename.replace('codebooks', 'oldcodes')
        with open(oldCodesFilename, 'wb') as f:
            pickle.dump(oldAssigns, f)
    
    return codebooks, assigns, oldAssigns

def onlineReBatch(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                trainN, inferenceN, codebooksFilename, codeFilename, \
                threadsCount=15, itsCount=20, \
                k=1024):
    pointsCount = points.shape[0]
    assigns = np.zeros((pointsCount, M), dtype='int32')
    codebooks = np.zeros((M, K, dim), dtype='float32')
    for it in range(60):
        baseEndIndex = 0
        ATAInv = np.zeros((M*K, M*K), dtype='float32')
        ATb = np.zeros((M*K, dim), dtype='float32')

        #Init points
        initEndIndex = startIndexPerBatch[1]
        initPoints = points[:initEndIndex,:]
        print(f'init end index is {initEndIndex}.')

        if it == 0:
            initCodebooks,initAssigns,initOldAssigns = learnCodebooksAQ(initPoints, dim, M, K, \
                                                        initEndIndex, codebooksFilename, \
                                                        trainN, saveCodebook=False, \
                                                        threadsCount=threadsCount, \
                                                        itsCount=1)
        else:
            initCodebooks,initAssigns,initOldAssigns = learnCodebooksAQ(initPoints, dim, M, K, \
                                                        initEndIndex, codebooksFilename, \
                                                        trainN, saveCodebook=False, \
                                                        threadsCount=threadsCount, \
                                                        itsCount=1,CODEBOOKS=codebooks, ASSIGNS=assigns[:initEndIndex,:])

        print(f'Begin to initalize.')

        codebooks[:] = initCodebooks
        assigns[:initEndIndex,:] = initAssigns

        data = np.ones(M * initEndIndex, dtype='float32')
        indices = np.zeros(M * initEndIndex, dtype='int32')
        indptr = np.array(range(0, initEndIndex + 1)) * M

        for i in range(initEndIndex * M):
                indices[i] = 0
        for pid in range(initEndIndex):
            for m in range(M):
                indices[pid * M + m] = m * K + initOldAssigns[pid,m]

        A = sparse.csr_matrix((data, indices, indptr), shape=(initEndIndex, M*K), copy=False).toarray().astype('float32')
        
        ATAInv[:] = np.linalg.pinv(np.dot(A.T, A))

        ATb[:] = A.T @ initPoints

        baseEndIndex = initEndIndex
        print(f'base end index is {baseEndIndex}.')
        print(f'finish initalize.')

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
            result = searchNearestNeighborsAQFAST(assigns[:baseEndIndex,:], codebooks, curBatchPoints,
                                            curBatchPointsNum, threadsCount=threadsCount, k=1024)
            end_t = time.time()
            print(f'search nearest neighbor time: {end_t-start_t}s.')

            start_t = time.time()
            getRecall(curBatchPoints, points[:baseEndIndex,:], result, threadsCount)
            end_t = time.time()
            print(f'Calculate recall rate time: {end_t-start_t}s.')

            
            #update codebook part
            start_t = time.time()
            (curBatchAssigns, curBatchErrors) = encodePointsAQ(curBatchPoints, codebooks, inferenceN, mode=encodeMode)
            end_t = time.time()
            print(f'Encode data points time: {end_t-start_t}s.')

            assigns[startIndex:curBatchEndIndex,:] = curBatchAssigns
            print(f'current batch data encode error is {np.mean(curBatchErrors)}.')

            cur_data = np.ones(M * curBatchPointsNum, dtype='float32')
            cur_indices = np.zeros(M * curBatchPointsNum, dtype='int32')
            cur_indptr = np.array(range(0, curBatchPointsNum + 1)) * M

            for i in range(curBatchPointsNum * M):
                    cur_indices[i] = 0
            for pid in range(curBatchPointsNum):
                for m in range(M):
                    cur_indices[pid * M + m] = m * K + curBatchAssigns[pid,m]

            start_t = time.time()
            Ai =  sparse.csr_matrix((cur_data, cur_indices, cur_indptr), shape=(curBatchPointsNum, M*K), copy=False).toarray().astype('float32')

            ATAInv[:] = ATAInv - ATAInv @ Ai.T @ np.linalg.inv(np.eye(curBatchPointsNum) + Ai @ ATAInv @ Ai.T) @ Ai @ ATAInv
            
            ATb = ATb + Ai.T @ curBatchPoints

            X = ATAInv @ ATb 
            codebooks[:] = X.reshape(M,K,-1)
            end_t = time.time()
            print(f'Update codebooks time {end_t - start_t}s.')

            baseEndIndex = curBatchEndIndex
        
        (curAssigns, curErrors) = encodePointsAQ(points, codebooks, trainN, mode=encodeMode)
        print(f'{it}/60:All points quanzation error is {np.mean(curErrors)}.')
        assigns[:] = curAssigns


def getQuantizationErrorAQ(testFilename, dim, pointsCount, codebooksFilename, branch):
    with open(codebooksFilename, 'rb') as f:
        codebooks = pickle.load(f)

    points = fvecs_read(testFilename)
    _, errors = encodePointsAQ(points, codebooks, branch, mode=encodeMode)
    return np.mean(errors)

def encodeDatasetAQ(baseFilename, pointsCount, codebooksFilename, codeFilename, branch):
    with open(baseFilename, 'rb') as f:
        codebooks = pickle.load(f)

    points = fvecs_read(baseFilename)
    (codes, errors) = encodePointsAQ(points, codebooks, branch, mode=encodeMode)
    print(f"Mean AQ quantization error: {np.mean(errors)}")
    with open(codeFilename, 'wb') as f:
        pickle.dump(codes, f)

def calculateDistancesAQ(qid, codebooksProducts, queryCodebookProducts, pointCodes, listLength):
    distances = np.zeros(pointCodes.shape[0], dtype='float32')
    for pid in range(pointCodes.shape[0]):
        for i in range(codebooksProducts.shape[0]):
            distances[pid] -= 2 * queryCodebookProducts[i, qid, pointCodes[pid, i]]
            distances[pid] += codebooksProducts[i, i, pointCodes[pid, i], pointCodes[pid, i]]
            for j in range(i+1, codebooksProducts.shape[1]):
               distances[pid] += 2 * codebooksProducts[i, j, pointCodes[pid, i], pointCodes[pid, j]]
    return distances.argsort()[0:listLength]

def searchNearestNeighborsAQ(codesSource, codebookSource, queriesSource, queriesCount, threadsCount=15, k=1024):
    """
        codeFilename: base_vector 的编码表
        codebookFilename：codebooks 的存放位置
        queriesFilename: 查询向量的路径
        queriesCount:查询向量的个数
        threadsCount: 进程数量
        k:最近的k个邻居
    """
    if isinstance(codebookSource, str):
        with open(codebookSource, 'rb') as f:
            codebooks = pickle.load(f)
    else:
        codebooks = codebookSource
    M = codebooks.shape[0]
    codebookSize = codebooks.shape[1]
    print(f'codebook shape is {codebooks.shape}.')

    if isinstance(codesSource, str):
        with open(codesSource, 'rb') as f:
            codes = pickle.load(f)
    else:
        codes = codesSource
    print(f'codes shape is {codes.shape}.')

    if isinstance(queriesSource, str):
        queries = fvecs_read(queriesSource)
    else:
        queries = queriesSource
    print(f'queries shape is {queries.shape}.')

    codebooksProducts = np.zeros((M, M, codebookSize, codebookSize),dtype='float32')
    for i in range(M):
        for j in range(M):
            codebooksProducts[i, j, :, :] = np.dot(codebooks[i,:,:], codebooks[j,:,:].T)
    queryCodebookProducts = np.zeros((M, queriesCount, codebookSize),dtype='float32')
    for i in range(M):
        queryCodebookProducts[i,:,:] = np.dot(queries, codebooks[i,:,:].T)
    k = min(k, codes.shape[0])

    nearest = np.zeros((queries.shape[0], k), dtype='int32')

    pool = Pool(processes=threadsCount)
    ans = pool.map(partial(calculateDistancesAQ, \
                           codebooksProducts=codebooksProducts, \
                           queryCodebookProducts=queryCodebookProducts, \
                           pointCodes=codes, \
                           listLength=k), range(0,queries.shape[0]))
    pool.close()
    pool.join()

    for i in range(queries.shape[0]):
        nearest[i,:] = ans[i].flatten()
    return nearest

def searchNearestNeighborsAQFAST(codesSource, codebookSource, queriesSource, queriesCount, threadsCount=15, k=1024):
    """
        codeFilename: base_vector 的编码表
        codebookFilename：codebooks 的存放位置
        queriesFilename: 查询向量的路径
        queriesCount:查询向量的个数
        threadsCount: 进程数量
        k:最近的k个邻居
    """
    if isinstance(codebookSource, str):
        with open(codebookSource, 'rb') as f:
            codebooks = pickle.load(f)
    else:
        codebooks = codebookSource
    M = codebooks.shape[0]
    codebookSize = codebooks.shape[1]
    # print(f'codebook shape is {codebooks.shape}.')

    if isinstance(codesSource, str):
        with open(codesSource, 'rb') as f:
            codes = pickle.load(f)
    else:
        codes = codesSource
    # print(f'codes shape is {codes.shape}.')

    if isinstance(queriesSource, str):
        queries = fvecs_read(queriesSource)
    else:
        queries = queriesSource
    # print(f'queries shape is {queries.shape}.')

    codebooksProducts = np.zeros((M, M, codebookSize, codebookSize),dtype='float32')
    for i in range(M):
        for j in range(M):
            codebooksProducts[i, j, :, :] = np.dot(codebooks[i,:,:], codebooks[j,:,:].T)
    queryCodebookProducts = np.zeros((M, queriesCount, codebookSize),dtype='float32')

    for i in range(M):
        queryCodebookProducts[i,:,:] = np.dot(queries, codebooks[i,:,:].T)
    k = min(k, codes.shape[0])

    distances = np.zeros((queries.shape[0], codes.shape[0]), dtype='float32')

    for i in range(M):
        distances -= 2 *  queryCodebookProducts[i, np.arange(queries.shape[0]).reshape(-1,1), codes[:,i].reshape(1,-1)]
    
    for i in range(M):
        distances += codebooksProducts[i, i, codes[:,i], codes[:,i]]
        for j in range(i+1, M):
            distances += 2 * codebooksProducts[i, j, codes[:,i], codes[:,j]]
    
    return distances.argsort()[0:k]

def searchNearestNeighborsAQStreaming(codes,  codebooks, curData, k=1024):
    M = codebooks.shape[0]
    codebookSize = codebooks.shape[1]

    codebooksProducts = np.zeros((M, M, codebookSize, codebookSize),dtype='float32')
    for i in range(M):
        for j in range(M):
            codebooksProducts[i, j, :, :] = np.dot(codebooks[i,:,:], codebooks[j,:,:].T)
    queryCodebookProducts = np.dot(codebooks, curData)

    k = min(k, codes.shape[0])

    distances = np.zeros(codes.shape[0], dtype='float32')
     
    for i in range(M):
        distances -= 2 * queryCodebookProducts[i, codes[:, i]]
        distances += codebooksProducts[i, i, codes[:, i], codes[:, i]]
        for j in range(i+1, M):
            distances += 2 * codebooksProducts[i, j, codes[:, i], codes[:, j]]
    return distances.argsort()[0:k]

def encodePointsAQStreaming(curData, codebooks, inferenceN, mode=encodeMode):
    M = codebooks.shape[0]
    K = codebooks.shape[1]

    # 每个codeword与其他所有codeword的内积的2倍
    codebooksProducts = np.zeros((M,K,M*K), dtype='float32')
    # 中间的辅助变量，用以记录每个codeword与其他所有codeword的内积的2倍
    fullProducts = np.zeros((M,K,M,K), dtype='float32')
    # 每个codeword与自身的内积
    codebooksNorms = np.zeros((M*K), dtype='float32')

    for m1 in range(M):
        for m2 in range(M):
            fullProducts[m1,:,m2,:] = 2 * np.dot(codebooks[m1,:,:], codebooks[m2,:,:].T)
        codebooksNorms[m1*K:(m1+1)*K] = fullProducts[m1,:,m1,:].diagonal() / 2
        codebooksProducts[m1,:,:] = np.reshape(fullProducts[m1,:,:,:], (K,M*K))
    
    queryProducts = np.zeros((1, M*K), dtype='float32')

    for m in range(M):
        queryProducts[0,m*K:(m+1)*K] = 2 * np.dot(codebooks[m], curData)
    
    if mode == 'beam_search':
        (assign, error) = encodePointsBeamSearch(0, 1, queryProducts, codebooksProducts, codebooksNorms, inferenceN)
    elif mode == 'beam_sample':
        (assign, error) = encodePointsSampleBeamSearch(0, 1, queryProducts, codebooksProducts, codebooksNorms, inferenceN)
    else:
        raise('Error!')

    error += np.dot(curData, curData.T)

    return assign.reshape(-1), error

def onlineBatch(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                trainN, inferenceN, codebooksFilename, codeFilename, \
                threadsCount=15, itsCount=20, \
                k=1024):

    pointsCount = points.shape[0]
    
    baseEndIndex = 0
    assigns = np.zeros((pointsCount, M), dtype='int32')
    codebooks = np.zeros((M, K, dim), dtype='float32')
    ATAInv = np.zeros((M*K, M*K), dtype='float32')
    ATb = np.zeros((M*K, dim), dtype='float32')

    #Init points
    initEndIndex = startIndexPerBatch[1]
    initPoints = points[:initEndIndex,:]
    print(f'init end index is {initEndIndex}.')

    initCodebooks,initAssigns,initOldAssigns = learnCodebooksAQ(points, dim, M, K, \
                                                 pointsCount, codebooksFilename, \
                                                 trainN, saveCodebook=True, \
                                                 threadsCount=threadsCount, \
                                                 itsCount=itsCount)
    return
    # with open(codebooksFilename, 'rb') as f:
    #     initCodebooks = pickle.load(f)
    
    # codesFilename = codebooksFilename.replace('codebooks','codes')
    # with open(codesFilename, 'rb') as f:
    #     initAssigns = pickle.load(f)
    
    # oldCodesFilename = codebooksFilename.replace('codebooks', 'oldcodes')
    # with open(oldCodesFilename, 'rb') as f:
    #     initOldAssigns = pickle.load(f)

    print(f'Begin to initalize.')

    codebooks[:] = initCodebooks
    assigns[:initEndIndex,:] = initAssigns

    data = np.ones(M * initEndIndex, dtype='float32')
    indices = np.zeros(M * initEndIndex, dtype='int32')
    indptr = np.array(range(0, initEndIndex + 1)) * M

    for i in range(initEndIndex * M):
            indices[i] = 0
    for pid in range(initEndIndex):
        for m in range(M):
            indices[pid * M + m] = m * K + initOldAssigns[pid,m]

    A = sparse.csr_matrix((data, indices, indptr), shape=(initEndIndex, M*K), copy=False).toarray().astype('float32')
    
    ATAInv[:] = np.linalg.pinv(np.dot(A.T, A))

    ATb[:] = A.T @ initPoints

    baseEndIndex = initEndIndex
    print(f'base end index is {baseEndIndex}.')
    print(f'finish initalize.')

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
        result = searchNearestNeighborsAQFAST(assigns[:baseEndIndex,:], codebooks, curBatchPoints,
                                          curBatchPointsNum, threadsCount=threadsCount, k=1024)
        end_t = time.time()
        print(f'search nearest neighbor time: {end_t-start_t}s.')

        start_t = time.time()
        getRecall(curBatchPoints, points[:baseEndIndex,:], result, threadsCount)
        end_t = time.time()
        print(f'Calculate recall rate time: {end_t-start_t}s.')

        
        #update codebook part
        start_t = time.time()
        (curBatchAssigns, curBatchErrors) = encodePointsAQ(curBatchPoints, codebooks, inferenceN, mode=encodeMode)
        end_t = time.time()
        print(f'Encode data points time: {end_t-start_t}s.')

        assigns[startIndex:curBatchEndIndex,:] = curBatchAssigns
        print(f'current batch data encode error is {np.mean(curBatchErrors)}.')

        cur_data = np.ones(M * curBatchPointsNum, dtype='float32')
        cur_indices = np.zeros(M * curBatchPointsNum, dtype='int32')
        cur_indptr = np.array(range(0, curBatchPointsNum + 1)) * M

        for i in range(curBatchPointsNum * M):
                cur_indices[i] = 0
        for pid in range(curBatchPointsNum):
            for m in range(M):
                cur_indices[pid * M + m] = m * K + curBatchAssigns[pid,m]

        start_t = time.time()
        Ai =  sparse.csr_matrix((cur_data, cur_indices, cur_indptr), shape=(curBatchPointsNum, M*K), copy=False).toarray().astype('float32')

        ATAInv[:] = ATAInv - ATAInv @ Ai.T @ np.linalg.inv(np.eye(curBatchPointsNum) + Ai @ ATAInv @ Ai.T) @ Ai @ ATAInv
        
        ATb = ATb + Ai.T @ curBatchPoints

        X = ATAInv @ ATb 
        codebooks[:] = X.reshape(M,K,-1)
        end_t = time.time()
        print(f'Update codebooks time {end_t - start_t}s.')

        baseEndIndex = curBatchEndIndex

def onlineBatchRLAQ(points, startIndexPerBatch, dim, M, K, numGroup, \
                    codebooks, initAssigns, initOldAssigns, path_to_best_model, \
                    threadsCount=15, itsCount=20, \
                    k=1024):

    state_dict = torch.load(path_to_best_model, map_location='cpu')

    pointsCount = points.shape[0]
    baseEndIndex = 0
    assigns = np.zeros((pointsCount, M), dtype='int32')
    
    ATAInv = np.zeros((M*K, M*K), dtype='float32')
    ATb = np.zeros((M*K, dim), dtype='float32')

    #Init points
    initEndIndex = startIndexPerBatch[1]
    initPoints = points[:initEndIndex,:]
    print(f'init end index is {initEndIndex}.')

    print(f'Begin to initalize.')

    assigns[:initEndIndex,:] = initAssigns

    data = np.ones(M * initEndIndex, dtype='float32')
    indices = np.zeros(M * initEndIndex, dtype='int32')
    indptr = np.array(range(0, initEndIndex + 1)) * M

    for i in range(initEndIndex * M):
            indices[i] = 0
    for pid in range(initEndIndex):
        for m in range(M):
            indices[pid * M + m] = m * K + initOldAssigns[pid,m]

    A = sparse.csr_matrix((data, indices, indptr), shape=(initEndIndex, M*K), copy=False).toarray().astype('float32')
    
    ATAInv[:] = np.linalg.pinv(np.dot(A.T, A))

    ATb[:] = A.T @ initPoints

    baseEndIndex = initEndIndex
    print(f'base end index is {baseEndIndex}.')
    print(f'finish initalize.')

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
        result = searchNearestNeighborsAQFAST(assigns[:baseEndIndex,:], codebooks, curBatchPoints,
                                          curBatchPointsNum, threadsCount=threadsCount, k=1024)
        end_t = time.time()
        print(f'search nearest neighbor time: {end_t-start_t}s.')

        start_t = time.time()
        getRecall(curBatchPoints, points[:baseEndIndex,:], result)
        end_t = time.time()
        print(f'Calculate recall rate time: {end_t-start_t}s.')

        model = TestPolicyNet(2*dim, dim, torch.from_numpy(codebooks), M, K, 64)
        model.load_state_dict(state_dict)
        model.eval()

        #update codebook part
        start_t = time.time()
        (curBatchAssigns, curBatchErrors) = encodePointsRL(curBatchPoints, codebooks, model)
        end_t = time.time()
        del model

        print(f'Encode data points time: {end_t-start_t}s.')

        assigns[startIndex:curBatchEndIndex,:] = curBatchAssigns
        print(f'current batch data encode error is {np.mean(curBatchErrors)}.')

        cur_data = np.ones(M * curBatchPointsNum, dtype='float32')
        cur_indices = np.zeros(M * curBatchPointsNum, dtype='int32')
        cur_indptr = np.array(range(0, curBatchPointsNum + 1)) * M

        for i in range(curBatchPointsNum * M):
                cur_indices[i] = 0
        for pid in range(curBatchPointsNum):
            for m in range(M):
                cur_indices[pid * M + m] = m * K + curBatchAssigns[pid,m]

        start_t = time.time()
        Ai =  sparse.csr_matrix((cur_data, cur_indices, cur_indptr), shape=(curBatchPointsNum, M*K), copy=False).toarray().astype('float32')

        ATAInv[:] = ATAInv - ATAInv @ Ai.T @ np.linalg.inv(np.eye(curBatchPointsNum) + Ai @ ATAInv @ Ai.T) @ Ai @ ATAInv
        
        ATb = ATb + Ai.T @ curBatchPoints

        X = ATAInv @ ATb 
        codebooks = X.reshape(M,K,-1)
        end_t = time.time()
        print(f'Update codebooks time {end_t - start_t}s.')

        baseEndIndex = curBatchEndIndex

def onlineBatchRLAQMain(data, param, numGroup, \
                        threadsCount=15, itsCount=20, \
                        k=1024):
    points = np.load(data.points)
    startIndexPerBatch = np.load(data.startIndex)

    with open(data.codebooks, mode='rb') as f:
        codebooks = pickle.load(f)
    
    with open(data.initAssigns, mode='rb') as f:
        initAssigns = pickle.load(f)
    
    with open(data.initOldAssigns, mode='rb') as f:
        initOldAssigns = pickle.load(f)
    
    onlineBatchRLAQ(points, startIndexPerBatch, data.dim, param.M, param.K, numGroup, \
                    codebooks, initAssigns, initOldAssigns, data.bestmodel, \
                    threadsCount, itsCount, \
                    k)

def onlineBatchNoUpdateAQ(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                        trainN, inferenceN, codebooksFilename, codeFilename, \
                        threadsCount=15, itsCount=20, \
                        k=1024):
    
    pointsCount = points.shape[0]
    
    baseEndIndex = 0
    assigns = np.zeros((pointsCount, M), dtype='int32')
    codebooks = np.zeros((M, K, dim), dtype='float32')

    #Init points
    initEndIndex = startIndexPerBatch[1]
    initPoints = points[:initEndIndex,:]
    print(f'Init end index is {initEndIndex}.')

    initCodebooks, initAssigns, _ = learnCodebooksAQ(initPoints, dim, M, K, \
                                                 initEndIndex, codebooksFilename, \
                                                 trainN, saveCodebook=False, \
                                                 threadsCount=threadsCount, \
                                                 itsCount=itsCount)

    codebooks[:] = initCodebooks
    assigns[:initEndIndex,:] = initAssigns
    
    baseEndIndex = initEndIndex

    for idx,startIndex in enumerate(startIndexPerBatch):
        if idx == 0:
            continue
        print(f'group:{idx+1}/{numGroup}, new data coming.')
        #query part
        if idx < numGroup - 1:
            curBatchEndIndex = min(startIndexPerBatch[idx+1], pointsCount)
        else:
            curBatchEndIndex = pointsCount
        print(f'baseEndIndex is {baseEndIndex}.')
        print(f'startIndex:{startIndex}, curBatchEndIndex:{curBatchEndIndex}.')

        curBatchPoints = points[startIndex:curBatchEndIndex,:]
        curBatchPointsNum = curBatchPoints.shape[0]
        print(f'current batch end index is {curBatchEndIndex}.')
        print(f'current batch point number is {curBatchPointsNum}.')

        start_t = time.time()
        result = searchNearestNeighborsAQFAST(assigns[:baseEndIndex,:], codebooks, curBatchPoints,
                                            curBatchPointsNum, threadsCount=threadsCount, k=1024)
        end_t = time.time()
        print(f'search nearest neighbor time: {end_t - start_t}s.')

        start_t = time.time()
        getRecall(curBatchPoints, points[:baseEndIndex,:], result)
        end_t = time.time()
        print(f'Calculate recall rate time: {end_t - start_t}s.')

        #update codebook part
        start_t = time.time()
        (curBatchAssigns, curBatchErrors) = encodePointsAQ(curBatchPoints, codebooks, inferenceN, mode=encodeMode)
        assigns[startIndex:curBatchEndIndex,:] = curBatchAssigns
        print(f'current batch data encode error is {np.mean(curBatchErrors)}.')
        end_t = time.time()
        print(f'Encode data points time: {end_t-start_t}s.')

        baseEndIndex = curBatchEndIndex

def onlineBatchRetrainAQ(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                        trainN, inferenceN, codebooksFilename, codeFilename, \
                        threadsCount=15, itsCount=20, \
                        k=1024):

    pointsCount = points.shape[0]
    
    baseEndIndex = 0
    assigns = np.zeros((pointsCount, M), dtype='int32')
    codebooks = np.zeros((M, K, dim), dtype='float32')

    #Init points
    initEndIndex = startIndexPerBatch[1]
    initPoints = points[:initEndIndex,:]
    print(f'init end index is {initEndIndex}.')

    initCodebooks,initAssigns,_ = learnCodebooksAQ(initPoints, dim, M, K, \
                                                 initEndIndex, codebooksFilename, \
                                                 trainN, saveCodebook=False, \
                                                 threadsCount=threadsCount, \
                                                 itsCount=itsCount)
    #initialize
    print(f'Begin to initalize.')

    codebooks[:] = initCodebooks
    assigns[:initEndIndex,:] = initAssigns

    baseEndIndex = initEndIndex
    print(f'base end index is {baseEndIndex}.')
    print(f'finish initalize.')

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
        result = searchNearestNeighborsAQFAST(assigns[:baseEndIndex,:], codebooks, curBatchPoints,
                                          curBatchPointsNum, threadsCount=threadsCount, k=1024)
        end_t = time.time()
        print(f'search nearest neighbor time: {end_t-start_t}s.')

        start_t = time.time()
        getRecall(curBatchPoints, points[:baseEndIndex,:], result)
        end_t = time.time()
        print(f'Calculate recall rate time: {end_t-start_t}s.')

        start_t = time.time()

        if idx+1 == numGroup:
            break

        retrain_codebooks, retrain_assigns,_  = learnCodebooksAQ(points[:curBatchEndIndex,:],dim,M,K, \
                                                                 curBatchEndIndex, codebooksFilename, \
                                                                 trainN, saveCodebook=False, \
                                                                 threadsCount=threadsCount, \
                                                                 itsCount=5, CODEBOOKS=codebooks, \
                                                                 ASSIGNS=assigns[:baseEndIndex,:])
        codebooks = retrain_codebooks.copy()
        assigns[:curBatchEndIndex,:] = retrain_assigns

        end_t = time.time()
        print(f'Retrain time {end_t - start_t}s.')

        baseEndIndex = curBatchEndIndex

def onlineStreamingAQ(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                    trainN, inferenceN, codebooksFilename, codeFilename, \
                    threadsCount=15, itsCount=20, \
                    k=1024):

    pointsCount = points.shape[0]
    
    baseEndIndex = 0
    assigns = np.zeros((pointsCount, M), dtype='int32')
    codebooks = np.zeros((M, K, dim), dtype='float32')
    ATAInv = np.zeros((M*K, M*K), dtype='float32')
    ATb = np.zeros((M*K, dim), dtype='float32')

    #Init points
    initEndIndex = startIndexPerBatch[1]
    initPoints = points[:initEndIndex,:]
    print(f'init end index is {initEndIndex}.')

    initCodebooks,initAssigns,initOldAssigns = learnCodebooksAQ(initPoints, dim, M, K, \
                                                 initEndIndex, codebooksFilename, \
                                                 trainN, saveCodebook=False, \
                                                 threadsCount=threadsCount, \
                                                 itsCount=itsCount)
    #initialize
    print(f'Begin to initalize.')

    codebooks[:] = initCodebooks
    assigns[:initEndIndex,:] = initAssigns

    data = np.ones(M * initEndIndex, dtype='float32')
    indices = np.zeros(M * initEndIndex, dtype='int32')
    indptr = np.array(range(0, initEndIndex + 1)) * M

    for i in range(initEndIndex * M):
            indices[i] = 0
    for pid in range(initEndIndex):
        for m in range(M):
            indices[pid * M + m] = m * K + initOldAssigns[pid,m]
    A = sparse.csr_matrix((data, indices, indptr), shape=(initEndIndex, M*K), copy=False).toarray().astype('float32')
    
    ATAInv[:] = np.linalg.pinv(np.dot(A.T, A))

    ATb[:] = A.T @ initPoints

    print(f'finish initalize.')

    for idx,startIndex in enumerate(startIndexPerBatch):
        if idx == 0:
            continue
        print(f'group:{idx+1}/{numGroup}, new data coming.')
        
        if idx < numGroup - 1:
            curBatchEndIndex = min(startIndexPerBatch[idx+1], pointsCount)
        else:
            curBatchEndIndex = pointsCount
            
        curBatchPoints = points[startIndex:curBatchEndIndex,:]
        curBatchPointsNum = curBatchPoints.shape[0]
        print(f'current batch end index is {curBatchEndIndex}.')
        print(f'current batch point number is {curBatchPointsNum}.')
        
        curRecall = np.zeros(12)
        curBatchError = 0
        for curIndex in range(startIndex, curBatchEndIndex):
            curData = points[curIndex]
            result = searchNearestNeighborsAQStreaming(assigns[:curIndex,:], codebooks, curData, k=k)
            curRecall += getRecallStreaming(curData, points[:curIndex], result, 11)

            (curAssign, curError) = encodePointsAQStreaming(curData, codebooks, inferenceN, mode=encodeMode)
            assigns[curIndex,:] = curAssign
            curBatchError += curError

            a = np.zeros((M*K, 1), dtype='float32')
            a[np.arange(M)*K+curAssign] = 1.0

            ATAInv = ATAInv - (ATAInv @ a @ a.T @ ATAInv)/(1+a.T @ ATAInv @ a)

            ATb = ATb + a @ curData.reshape(1,-1)

            X = ATAInv @ ATb

            codebooks[:] = X.reshape(M,K,-1)
        
        T = [2**i for i in range(11)]
        T.insert(5, 20)

        for idx, t in enumerate(T):
            print(f'Recall@{t} is {curRecall[idx]/(curBatchPointsNum)}.')
        print(f'Current batch quantation error is {curBatchError/curBatchPointsNum}.')

def onlineStreamingBatchAQ(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                            trainN, inferenceN, codebooksFilename, codeFilename, \
                            threadsCount=15, itsCount=20, \
                            k=1024):

    pointsCount = points.shape[0]
    
    baseEndIndex = 0
    assigns = np.zeros((pointsCount, M), dtype='int32')
    codebooks = np.zeros((M, K, dim), dtype='float32')
    ATAInv = np.zeros((M*K, M*K), dtype='float32')
    ATb = np.zeros((M*K, dim), dtype='float32')

    #Init points
    initEndIndex = startIndexPerBatch[1]
    initPoints = points[:initEndIndex,:]
    print(f'init end index is {initEndIndex}.')

    # initCodebooks,initAssigns,initOldAssigns = learnCodebooksAQ(initPoints, dim, M, K, \
    #                                              initEndIndex, codebooksFilename, \
    #                                              trainN, saveCodebook=False, \
    #                                              threadsCount=threadsCount, \
    #                                              itsCount=itsCount)
    with open(codebooksFilename, 'rb') as f:
        initCodebooks = pickle.load(f)
    
    codesFilename = codebooksFilename.replace('codebooks','codes')
    with open(codesFilename, 'rb') as f:
        initAssigns = pickle.load(f)
    
    oldCodesFilename = codebooksFilename.replace('codebooks', 'oldcodes')
    with open(oldCodesFilename, 'rb') as f:
        initOldAssigns = pickle.load(f)
    #initialize
    print(f'Begin to initalize.')

    codebooks[:] = initCodebooks
    assigns[:initEndIndex,:] = initAssigns

    data = np.ones(M * initEndIndex, dtype='float32')
    indices = np.zeros(M * initEndIndex, dtype='int32')
    indptr = np.array(range(0, initEndIndex + 1)) * M

    for i in range(initEndIndex * M):
            indices[i] = 0
    for pid in range(initEndIndex):
        for m in range(M):
            indices[pid * M + m] = m * K + initOldAssigns[pid,m]
    A = sparse.csr_matrix((data, indices, indptr), shape=(initEndIndex, M*K), copy=False).toarray().astype('float32')
    
    ATAInv[:] = np.linalg.pinv(np.dot(A.T, A))

    ATb[:] = A.T @ initPoints

    print(f'finish initalize.')

    for idx,startIndex in enumerate(startIndexPerBatch):
        if idx == 0:
            continue
        print(f'group:{idx+1}/{numGroup}, new data coming.')
        
        if idx < numGroup - 1:
            curBatchEndIndex = min(startIndexPerBatch[idx+1], pointsCount)
        else:
            curBatchEndIndex = pointsCount
            
        curBatchPoints = points[startIndex:curBatchEndIndex,:]
        curBatchPointsNum = curBatchPoints.shape[0]
        print(f'current batch end index is {curBatchEndIndex}.')
        print(f'current batch point number is {curBatchPointsNum}.')
        
        step_size = 512
        curRecall = np.zeros(12)
        curBatchError = 0

        for curIndex in range(startIndex, curBatchEndIndex, step_size):
            endIndex = min(curIndex+step_size, curBatchEndIndex)

            minStepPointsNum = endIndex - curIndex

            curData = points[curIndex:endIndex,:]

            result = searchNearestNeighborsAQFAST(assigns[:curIndex,:], codebooks, curData,
                                          minStepPointsNum, threadsCount=threadsCount, k=1024)
            
            curRecall += getRecallBatch(curData, points[:curIndex,:], result, threadsCount)

            (curAssign, curError) = encodePointsAQ(curData, codebooks, inferenceN, mode=encodeMode)

            assigns[curIndex:endIndex,:] = curAssign
            
            curBatchError += np.sum(curError)

            cur_data = np.ones(M * minStepPointsNum, dtype='float32')
            cur_indices = np.zeros(M * minStepPointsNum, dtype='int32')
            cur_indptr = np.array(range(0, minStepPointsNum + 1)) * M

            for i in range(minStepPointsNum * M):
                    cur_indices[i] = 0
            for pid in range(minStepPointsNum):
                for m in range(M):
                    cur_indices[pid * M + m] = m * K + curAssign[pid,m]

            Ai =  sparse.csr_matrix((cur_data, cur_indices, cur_indptr), shape=(minStepPointsNum, M*K), copy=False).toarray().astype('float32')

            ATAInv[:] = ATAInv - ATAInv @ Ai.T @ np.linalg.inv(np.eye(minStepPointsNum) + Ai @ ATAInv @ Ai.T) @ Ai @ ATAInv
            
            ATb = ATb + Ai.T @ curData

            X = ATAInv @ ATb 
            codebooks[:] = X.reshape(M,K,-1)
        
        T = [2**i for i in range(11)]
        T.insert(5, 20)

        for idx, t in enumerate(T):
            print(f'Recall@{t} is {curRecall[idx]/(curBatchPointsNum)}.')
        print(f'Current batch quantation error is {curBatchError/curBatchPointsNum}.')

def searchNearestNeighborsOSH(baseAssigns, curBatchAssigns, k=1024):
    k = min(baseAssigns.shape[0], k)
    queryNum = curBatchAssigns.shape[0]

    result = np.zeros((queryNum, k), dtype=np.int32)

    for i in range(queryNum):
        hamming_dis = np.logical_xor(baseAssigns, curBatchAssigns[i]).sum(1)
        result[i,:] = np.argsort(hamming_dis)[:k]
    
    return result


L = 200

"""
    Sketching Size
"""
def matrixSketching(BMatrix, APoints):
    if L%2 != 0 :
        raise(f"l must be even number.")
    
    ind = L // 2

    n = APoints.shape[0]
    numNonzeroRows = (np.sum(BMatrix**2, axis=1)>0).sum()

    for i in range(n):
        if numNonzeroRows < L:
            BMatrix[numNonzeroRows,:] = APoints[i,:]
            numNonzeroRows += 1
        else:
            q, r = np.linalg.qr(BMatrix.T)

            u, sigma, _ = np.linalg.svd(r)

            v = q @ u
            sigmaSquare = sigma ** 2
            theta = sigmaSquare[ind]
            sigmaHat = np.sqrt(np.maximum(np.diag(sigmaSquare) - np.eye(L)*theta, 0))
            BMatrix = sigmaHat @ v.T

            numNonzeroRows = ind

            BMatrix[numNonzeroRows,:] = APoints[i,:]
    
    return BMatrix

def SRHT(H, l ,m):
    q = l // 2
    sample = np.random.permutation(np.array(list(range(m))))

    I_m = np.eye(m)

    S = np.sqrt(m/q)*I_m[sample[:q],:]

    x = (np.random.rand(m)<0.5)*2-1

    D = np.diag(x)

    return S @ H @ D

def matrixSketchingFFD(BMatrix, F, APoints, H, m=512):
    if L%2 != 0 :
        raise(f"L must be even number.")
    
    ind = L // 2
    n = APoints.shape[0]
    d = APoints.shape[1]

    numNonzeroRows = (np.sum(BMatrix**2, axis=1)>0).sum()
    if numNonzeroRows == 0:
        F = np.zeros((m, d))

    for i in range(n):
        if numNonzeroRows < m:
            F[numNonzeroRows,:] = APoints[i,:]
            numNonzeroRows += 1
        else:
            # F[numNonzeroRows,:] = APoints[i,:]

            phi = SRHT(H, L, m)
            BMatrix[ind:,:] = phi @ F

            q, r = np.linalg.qr(BMatrix.T)

            u, sigma, _ = np.linalg.svd(r)

            v = q @ u
            sigmaSquare = sigma ** 2
            theta = sigmaSquare[ind]
            sigmaHat = np.sqrt(np.maximum(np.diag(sigmaSquare) - np.eye(L)*theta, 0))
            BMatrix = sigmaHat @ v.T

            numNonzeroRows = 0
            F = np.zeros((m, d))
    
    return BMatrix, F

mm = 512
def onlineBatchOSH(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                    trainN, inferenceN, codebooksFilename, codeFilename, \
                    threadsCount=15, itsCount=20, \
                    k=1024):

    pointsCount = points.shape[0]
    
    baseEndIndex = 0
    assigns = np.zeros((pointsCount, M), dtype='int32')

    Y = np.zeros((L, dim))
    mean_u = 0.0
    num_n = 0

    hashMatrix =np.random.rand(dim, M) - 0.5

    #Init points
    initEndIndex = startIndexPerBatch[1]
    initPoints = points[:initEndIndex,:]
    print(f'init end index is {initEndIndex}.')

    initPointsMean = np.mean(initPoints, axis=0) 
    F = np.zeros((mm, 512))
    H = np.sqrt(1/mm)*scipy.linalg.hadamard(mm)

    Y,F = matrixSketchingFFD(Y, F, initPoints - initPointsMean, H, mm)
    num_n = initEndIndex
    mean_u = initPointsMean

    q,r = np.linalg.qr(Y.T)
    u,_,_ = np.linalg.svd(r)
    v = q @ u

    hashProjMatOrg = v[:, 0:M]

    R = ortho_group.rvs(M)

    hashMatrix[:] = hashProjMatOrg @ R

    initAssigns = (initPoints - initPointsMean) @ hashMatrix

    assigns[:initEndIndex,:] = (initAssigns>0)
    
    baseEndIndex = initEndIndex
    print(f'base end index is {baseEndIndex}.')
    print(f'finish initalize.')

    for idx,startIndex in enumerate(startIndexPerBatch):
        if idx == 0:
            continue
        print(f'group:{idx+1}/{numGroup}, new data coming.')

        if idx < numGroup - 1:
            curBatchEndIndex = min(startIndexPerBatch[idx+1], pointsCount)
        else:
            curBatchEndIndex = pointsCount
            
        curBatchPoints = points[startIndex:curBatchEndIndex,:]
        curBatchPointsNum = curBatchPoints.shape[0]
        print(f'current batch end index is {curBatchEndIndex}.')
        print(f'current batch point number is {curBatchPointsNum}.')
        curBatchPointsMean = np.mean(curBatchPoints, axis=0)

        # curBatchInnerProduct = (curBatchPoints- mean_u) @ hashMatrix
        # curBatchAssigns = curBatchInnerProduct > 0

        # result = searchNearestNeighborsOSH(assigns[:baseEndIndex], curBatchAssigns, k=k)

        # getRecall(curBatchPoints, points[:baseEndIndex,:], result, threadsCount)

        # assigns[startIndex:curBatchEndIndex,:] = curBatchAssigns

        #query part
        
        sqrtNum = np.sqrt((num_n*curBatchPointsNum)/(num_n+curBatchPointsNum))

        A =np.vstack((curBatchPoints-curBatchPointsMean, sqrtNum*(curBatchPointsMean-mean_u)))

        Y,F = matrixSketchingFFD(Y, F, curBatchPoints-curBatchPointsMean, H, mm)

        mean_u = (num_n*mean_u + curBatchPointsNum*curBatchPointsMean)/(num_n + curBatchPointsNum)
        num_n  = num_n + curBatchPointsNum

        q,r = np.linalg.qr(Y.T)
        u,_,_ = np.linalg.svd(r)
        v = q @ u

        hashProjMatOrg = v[:, 0:M]

        R = ortho_group.rvs(M)

        hashMatrix[:] = hashProjMatOrg @ R

        assigns[:curBatchEndIndex] = ((points[:curBatchEndIndex]-mean_u) @ hashMatrix) > 0

        result = searchNearestNeighborsOSH(assigns[:baseEndIndex], assigns[startIndex:curBatchEndIndex], k=k)

        getRecall(curBatchPoints, points[:baseEndIndex,:], result, threadsCount)

        baseEndIndex = curBatchEndIndex

lamda = 0.5
"""
    regularization parameter
"""

def onlineStreamingSlidingWindowAQ(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                                trainN, inferenceN, codebooksFilename, codeFilename, \
                                threadsCount=15, itsCount=20, \
                                k=1000):
    
    pointsCount = points.shape[0]
    baseEndIndex = 0
    assigns = np.zeros((pointsCount, M), dtype='int32')
    codebooks = np.zeros((M, K, dim), dtype='float32')
    ATAInv = np.zeros((M*K, M*K), dtype='float32')
    ATb = np.zeros((M*K, dim), dtype='float32')

    initEndIndex = startIndexPerBatch[1]
    initPoints = points[:initEndIndex,:]
    print(f'init end index is {initEndIndex}.')

    initCodebooks,initAssigns,initOldAssigns = learnCodebooksAQ(initPoints, dim, M, K, \
                                                 initEndIndex, codebooksFilename, \
                                                 trainN, saveCodebook=False, \
                                                 threadsCount=threadsCount, \
                                                 itsCount=itsCount)

    # with open(codebooksFilename, 'rb') as f:
    #     initCodebooks = pickle.load(f)
    
    # codesFilename = codebooksFilename.replace('codebooks','codes')
    # with open(codesFilename, 'rb') as f:
    #     initAssigns = pickle.load(f)
    
    # oldCodesFilename = codebooksFilename.replace('codebooks', 'oldcodes')
    # with open(oldCodesFilename, 'rb') as f:
    #     initOldAssigns = pickle.load(f)


    print(f'Begin to initialize.')
    codebooks[:] = initCodebooks
    assigns[:initEndIndex,:] = initAssigns

    data = np.ones(M * initEndIndex, dtype='float32')
    indices = np.zeros(M * initEndIndex, dtype='int32')
    indptr = np.array(range(0, initEndIndex + 1)) * M

    for i in range(initEndIndex * M):
            indices[i] = 0
    for pid in range(initEndIndex):
        for m in range(M):
            indices[pid * M + m] = m * K + initOldAssigns[pid,m]
    A = sparse.csr_matrix((data, indices, indptr), shape=(initEndIndex, M*K), copy=False).toarray().astype('float32')
    
    ATAInv[:] = np.linalg.pinv(np.dot(A.T, A) + lamda * np.eye(M*K))

    ATb[:] = A.T @ initPoints
    
    #init sliding window
    window = deque(maxlen=winSize)

    window.extend(random.sample(list(range(initEndIndex)), winSize))
    # window.extend(list(range(initEndIndex))[:winSize])

    print(f'finish initalize.')
    
    for idx, startIndex in enumerate(startIndexPerBatch):
        if idx == 0:
            continue
        print(f'group:{idx+1}/{numGroup}, new data coming.')
        
        if idx < numGroup - 1:
            curBatchEndIndex = min(startIndexPerBatch[idx+1], pointsCount)
        else:
            curBatchEndIndex = pointsCount
        
        curBatchPoints = points[startIndex:curBatchEndIndex,:]
        curBatchPointsNum = curBatchPoints.shape[0]
        print(f'Current batch end index is {curBatchEndIndex}.')
        print(f'Current batch point number is {curBatchPointsNum}.')

        curRecall = np.zeros(11)
        curBatchError = 0

        for newIndex in range(startIndex, curBatchEndIndex):
            #new data coming
            newData = points[newIndex]
            
            #query part 
            #To be changed
            winIndex = np.array(window, dtype=np.int32)
            result = searchNearestNeighborsAQStreaming(assigns[winIndex,:], codebooks, newData, k)
            res = getRecallStreaming(newData, points[winIndex,:], result, 10)
            curRecall += res
            
            # print(f'{newIndex}:Current point Recall is {res}.')
            #update part
            oldIndex = window.popleft()
            window.append(newIndex)
            #update codebook based on new data
            (newAssign, newError) = encodePointsAQStreaming(newData, codebooks, inferenceN)
            assigns[newIndex,:] = newAssign
            curBatchError += newError

            a = np.zeros((M*K, 1), dtype='float32')
            a[np.arange(M)*K + newAssign] = 1.0

            # # print(f'{newIndex}:new data: {1+a.T @ ATAInv @ a}.')
            ATAInv = ATAInv - (ATAInv @ a @ a.T @ ATAInv)/(1+a.T @ ATAInv @ a)

            ATb = ATb + a @ newData.reshape(1,-1)

            X = ATAInv @ ATb

            codebooks[:] = X.reshape(M,K,-1)
            
            #update codebook based on old data
            # (oldAssign, oldError) = encodePointsAQStreaming(oldData, codebooks, inferenceN)
            
            oldData = points[oldIndex]
            oldAssign = assigns[oldIndex]
            b = np.zeros((M*K, 1), dtype='float32')
            b[np.arange(M)*K + oldAssign] = 1.0

            num = 1 - b.T @ ATAInv @ b
            # print(f'{newIndex}:old data: {1 - b.T @ ATAInv @ b}.')
            if abs(num.item()) >= 0.15:
                ATAInv = ATAInv + (ATAInv @ b @ b.T @ ATAInv)/(1 - b.T @ ATAInv @ b)

                ATb = ATb - b @ oldData.reshape(1,-1)

                X = ATAInv @ ATb

                codebooks[:] = X.reshape(M,K,-1)

        T = [2**i for i in range(10)]
        T.insert(5, 20)

        for idx, t in enumerate(T):
            print(f'Recall@{t} {curRecall[idx]/(curBatchPointsNum)}.')
        print(f'Current batch quantation error is {curBatchError/curBatchPointsNum}.')

def onlineBatchSlidingWindowAQ(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                            trainN, inferenceN, codebooksFilename, codeFilename, \
                            threadsCount=15, itsCount=20, \
                            k=1000):

    pointsCount = points.shape[0]
    
    baseEndIndex = 0
    assigns = np.zeros((pointsCount, M), dtype='int32')
    codebooks = np.zeros((M, K, dim), dtype='float32')
    ATAInv = np.zeros((M*K, M*K), dtype='float32')
    ATb = np.zeros((M*K, dim), dtype='float32')

    #Init points
    initEndIndex = startIndexPerBatch[1]
    initPoints = points[:initEndIndex,:]
    print(f'init end index is {initEndIndex}.')

    # initCodebooks,initAssigns,initOldAssigns = learnCodebooksAQ(initPoints, dim, M, K, \
    #                                              initEndIndex, codebooksFilename, \
    #                                              trainN, saveCodebook=True, \
    #                                              threadsCount=threadsCount, \
    #                                              itsCount=itsCount)
    with open(codebooksFilename, 'rb') as f:
        initCodebooks = pickle.load(f)
    
    codesFilename = codebooksFilename.replace('codebooks','codes')
    with open(codesFilename, 'rb') as f:
        initAssigns = pickle.load(f)
    
    oldCodesFilename = codebooksFilename.replace('codebooks', 'oldcodes')
    with open(oldCodesFilename, 'rb') as f:
        initOldAssigns = pickle.load(f)

    print(f'Begin to initalize.')

    codebooks[:] = initCodebooks
    assigns[:initEndIndex,:] = initAssigns

    data = np.ones(M * initEndIndex, dtype='float32')
    indices = np.zeros(M * initEndIndex, dtype='int32')
    indptr = np.array(range(0, initEndIndex + 1)) * M

    for i in range(initEndIndex * M):
            indices[i] = 0
    for pid in range(initEndIndex):
        for m in range(M):
            indices[pid * M + m] = m * K + initOldAssigns[pid,m]

    A = sparse.csr_matrix((data, indices, indptr), shape=(initEndIndex, M*K), copy=False).toarray().astype('float32')
    
    ATAInv[:] = np.linalg.pinv(np.dot(A.T, A) + lamda * np.eye(M*K))

    ATb[:] = A.T @ initPoints

    baseEndIndex = initEndIndex
    print(f'base end index is {baseEndIndex}.')

    #init sliding window
    window = deque(maxlen=winSize)
    window.extend(random.sample(list(range(initEndIndex)), min(winSize, initEndIndex)))
    
    print(f'finish initalize.')

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
        result = searchNearestNeighborsAQFAST(assigns[winIndex,:], codebooks, curBatchPoints,
                                          curBatchPointsNum, threadsCount=threadsCount, k=k)
        end_t = time.time()
        print(f'search nearest neighbor time: {end_t-start_t}s.')

        start_t = time.time()
        getRecall(curBatchPoints, points[winIndex,:], result, threadsCount)
        end_t = time.time()
        print(f'Calculate recall rate time: {end_t-start_t}s.')
        
        #---------------------------------------------------------------------#

        # #update codebook based on new data
        start_t = time.time()
        (curBatchAssigns, curBatchErrors) = encodePointsAQ(curBatchPoints, codebooks, inferenceN, mode=encodeMode)
        end_t = time.time()
        print(f'Encode data points time: {end_t-start_t}s.')

        oldBatchPointsNum = min(curBatchPointsNum, winSize)
        oldIndex = winIndex[:oldBatchPointsNum]
        window.extend(list(range(startIndex, curBatchEndIndex)))

        # # #update codebooks based on old data
        oldBatchData = points[oldIndex,:]
        oldBatchAssigns = assigns[oldIndex,:]
        start_t = time.time()
        # (oldBatchAssigns, oldBatchErrors) = encodePointsAQ(oldBatchData, codebooks, inferenceN, mode=encodeMode)
        end_t = time.time()
        print(f'Encode data points time: {end_t-start_t}s.')

        cur_data = np.ones(M * oldBatchPointsNum, dtype='float32')
        cur_indices = np.zeros(M * oldBatchPointsNum, dtype='int32')
        cur_indptr = np.array(range(0, oldBatchPointsNum + 1)) * M

        for i in range(oldBatchPointsNum * M):
                cur_indices[i] = 0
        for pid in range(oldBatchPointsNum):
            for m in range(M):
                cur_indices[pid * M + m] = m * K + oldBatchAssigns[pid,m]

        start_t = time.time()
        Ai =  sparse.csr_matrix((cur_data, cur_indices, cur_indptr), shape=(oldBatchPointsNum, M*K), copy=False).toarray().astype('float32')
        
        ATAInv[:] = ATAInv + ATAInv @ Ai.T @ np.linalg.inv(np.eye(oldBatchPointsNum) - Ai @ ATAInv @ Ai.T) @ Ai @ ATAInv
        
        ATb = ATb - Ai.T @ oldBatchData

        X = ATAInv @ ATb 
        codebooks[:] = X.reshape(M,K,-1)
        end_t = time.time()
        print(f'Update codebooks time {end_t - start_t}s.')
        
        #update codebooks based on new data
        assigns[startIndex:curBatchEndIndex,:] = curBatchAssigns
        print(f'current batch data encode error is {np.mean(curBatchErrors)}.')

        cur_data = np.ones(M * curBatchPointsNum, dtype='float32')
        cur_indices = np.zeros(M * curBatchPointsNum, dtype='int32')
        cur_indptr = np.array(range(0, curBatchPointsNum + 1)) * M

        for i in range(curBatchPointsNum * M):
                cur_indices[i] = 0
        for pid in range(curBatchPointsNum):
            for m in range(M):
                cur_indices[pid * M + m] = m * K + curBatchAssigns[pid,m]

        start_t = time.time()
        Ai =  sparse.csr_matrix((cur_data, cur_indices, cur_indptr), shape=(curBatchPointsNum, M*K), copy=False).toarray().astype('float32')

        ATAInv[:] = ATAInv - ATAInv @ Ai.T @ np.linalg.inv(np.eye(curBatchPointsNum) + Ai @ ATAInv @ Ai.T) @ Ai @ ATAInv
        
        ATb = ATb + Ai.T @ curBatchPoints

        X = ATAInv @ ATb 
        codebooks[:] = X.reshape(M,K,-1)
        end_t = time.time()
        print(f'Update codebooks time {end_t - start_t}s.')

        #update codebooks based on new data again
        start_t = time.time()
        (curBatchAssigns, curBatchErrors) = encodePointsAQ(curBatchPoints, codebooks, inferenceN, mode=encodeMode)
        end_t = time.time()
        print(f'Encode data points time: {end_t-start_t}s.')
        assigns[startIndex:curBatchEndIndex,:] = curBatchAssigns
        print(f'current batch data encode error is {np.mean(curBatchErrors)}.')

        cur_data = np.ones(M * curBatchPointsNum, dtype='float32')
        cur_indices = np.zeros(M * curBatchPointsNum, dtype='int32')
        cur_indptr = np.array(range(0, curBatchPointsNum + 1)) * M

        for i in range(curBatchPointsNum * M):
                cur_indices[i] = 0
        for pid in range(curBatchPointsNum):
            for m in range(M):
                cur_indices[pid * M + m] = m * K + curBatchAssigns[pid,m]

        start_t = time.time()
        Ai =  sparse.csr_matrix((cur_data, cur_indices, cur_indptr), shape=(curBatchPointsNum, M*K), copy=False).toarray().astype('float32')

        ATAInv[:] = ATAInv - ATAInv @ Ai.T @ np.linalg.inv(np.eye(curBatchPointsNum) + Ai @ ATAInv @ Ai.T) @ Ai @ ATAInv
        
        ATb = ATb + Ai.T @ curBatchPoints

        X = ATAInv @ ATb 
        codebooks[:] = X.reshape(M,K,-1)
        end_t = time.time()
        print(f'Update codebooks time {end_t - start_t}s.')
        #---------------------------------------------------------------------#

        #update codebook based on new data
        
        # start_t = time.time()
        # (curBatchAssigns, curBatchErrors) = encodePointsAQ(curBatchPoints, codebooks, inferenceN, mode=encodeMode)
        # end_t = time.time()
        # print(f'Encode data points time: {end_t-start_t}s.')

        # assigns[startIndex:curBatchEndIndex,:] = curBatchAssigns
        # print(f'current batch data encode error is {np.mean(curBatchErrors)}.')

        # cur_data = np.ones(M * curBatchPointsNum, dtype='float32')
        # cur_indices = np.zeros(M * curBatchPointsNum, dtype='int32')
        # cur_indptr = np.array(range(0, curBatchPointsNum + 1)) * M

        # for i in range(curBatchPointsNum * M):
        #         cur_indices[i] = 0
        # for pid in range(curBatchPointsNum):
        #     for m in range(M):
        #         cur_indices[pid * M + m] = m * K + curBatchAssigns[pid,m]

        # start_t = time.time()
        # Ai =  sparse.csr_matrix((cur_data, cur_indices, cur_indptr), shape=(curBatchPointsNum, M*K), copy=False).toarray().astype('float32')

        # ATAInv[:] = ATAInv - ATAInv @ Ai.T @ np.linalg.inv(np.eye(curBatchPointsNum) + Ai @ ATAInv @ Ai.T) @ Ai @ ATAInv
        
        # ATb = ATb + Ai.T @ curBatchPoints

        # X = ATAInv @ ATb 
        # codebooks[:] = X.reshape(M,K,-1)
        # end_t = time.time()
        # print(f'Update codebooks time {end_t - start_t}s.')

        # # #update window
        # oldBatchPointsNum = min(curBatchPointsNum, winSize)
        # oldIndex = winIndex[:oldBatchPointsNum]
        # window.extend(list(range(startIndex, curBatchEndIndex)))

        # #update codebooks based on old data
        # oldBatchData = points[oldIndex,:]
        # oldBatchAssigns = assigns[oldIndex,:]
        # start_t = time.time()
        # # (oldBatchAssigns, oldBatchErrors) = encodePointsAQ(oldBatchData, codebooks, inferenceN, mode=encodeMode)
        # end_t = time.time()
        # print(f'Encode data points time: {end_t-start_t}s.')

        # cur_data = np.ones(M * oldBatchPointsNum, dtype='float32')
        # cur_indices = np.zeros(M * oldBatchPointsNum, dtype='int32')
        # cur_indptr = np.array(range(0, oldBatchPointsNum + 1)) * M

        # for i in range(oldBatchPointsNum * M):
        #         cur_indices[i] = 0
        # for pid in range(oldBatchPointsNum):
        #     for m in range(M):
        #         cur_indices[pid * M + m] = m * K + oldBatchAssigns[pid,m]

        # start_t = time.time()
        # Ai =  sparse.csr_matrix((cur_data, cur_indices, cur_indptr), shape=(oldBatchPointsNum, M*K), copy=False).toarray().astype('float32')
        
        # ATAInv[:] = ATAInv + ATAInv @ Ai.T @ np.linalg.inv(np.eye(oldBatchPointsNum) - Ai @ ATAInv @ Ai.T) @ Ai @ ATAInv
        
        # ATb = ATb - Ai.T @ oldBatchData

        # X = ATAInv @ ATb 
        # codebooks[:] = X.reshape(M,K,-1)
        # end_t = time.time()
        # print(f'Update codebooks time {end_t - start_t}s.')
        
        baseEndIndex = curBatchEndIndex

def onlineRetrainSlidingWindowAQ(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                            trainN, inferenceN, codebooksFilename, codeFilename, \
                            threadsCount=15, itsCount=20, \
                            k=1000):

    pointsCount = points.shape[0]
    
    baseEndIndex = 0
    assigns = np.zeros((pointsCount, M), dtype='int32')
    codebooks = np.zeros((M, K, dim), dtype='float32')
    ATAInv = np.zeros((M*K, M*K), dtype='float32')
    ATb = np.zeros((M*K, dim), dtype='float32')

    #Init points
    initEndIndex = startIndexPerBatch[1]
    initPoints = points[:initEndIndex,:]
    print(f'init end index is {initEndIndex}.')

    initCodebooks,initAssigns,initOldAssigns = learnCodebooksAQ(initPoints, dim, M, K, \
                                                 initEndIndex, codebooksFilename, \
                                                 trainN, saveCodebook=True, \
                                                 threadsCount=threadsCount, \
                                                 itsCount=itsCount)
    # with open(codebooksFilename, 'rb') as f:
    #     initCodebooks = pickle.load(f)
    
    # codesFilename = codebooksFilename.replace('codebooks','codes')
    # with open(codesFilename, 'rb') as f:
    #     initAssigns = pickle.load(f)
    
    # oldCodesFilename = codebooksFilename.replace('codebooks', 'oldcodes')
    # with open(oldCodesFilename, 'rb') as f:
    #     initOldAssigns = pickle.load(f)

    print(f'Begin to initalize.')

    codebooks[:] = initCodebooks
    assigns[:initEndIndex,:] = initAssigns

    data = np.ones(M * initEndIndex, dtype='float32')
    indices = np.zeros(M * initEndIndex, dtype='int32')
    indptr = np.array(range(0, initEndIndex + 1)) * M

    for i in range(initEndIndex * M):
            indices[i] = 0
    for pid in range(initEndIndex):
        for m in range(M):
            indices[pid * M + m] = m * K + initOldAssigns[pid,m]

    A = sparse.csr_matrix((data, indices, indptr), shape=(initEndIndex, M*K), copy=False).toarray().astype('float32')
    
    ATAInv[:] = np.linalg.pinv(np.dot(A.T, A))

    ATb[:] = A.T @ initPoints

    baseEndIndex = initEndIndex
    print(f'base end index is {baseEndIndex}.')

    #init sliding window
    window = deque(maxlen=winSize)
    window.extend(random.sample(list(range(initEndIndex)), winSize))
    
    print(f'finish initalize.')

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
        result = searchNearestNeighborsAQFAST(assigns[:baseEndIndex,:], codebooks, curBatchPoints,
                                          curBatchPointsNum, threadsCount=threadsCount, k=k)
        end_t = time.time()
        print(f'search nearest neighbor time: {end_t-start_t}s.')

        start_t = time.time()
        getRecall(curBatchPoints, points[:baseEndIndex,:], result, threadsCount)
        end_t = time.time()
        print(f'Calculate recall rate time: {end_t-start_t}s.')

        #update codebook based on new data
        start_t = time.time()
        (curBatchAssigns, curBatchErrors) = encodePointsAQ(curBatchPoints, codebooks, inferenceN, mode=encodeMode)
        end_t = time.time()
        print(f'Encode data points time: {end_t-start_t}s.')

        assigns[startIndex:curBatchEndIndex,:] = curBatchAssigns
        print(f'current batch data encode error is {np.mean(curBatchErrors)}.')

        #update window
        oldIndex = winIndex[:curBatchPointsNum]
        window.extend(list(range(startIndex, curBatchEndIndex)))

        winIndex = np.array(window, dtype=np.int32)
        winPoints = points[winIndex]

        winCodebooks, winAssigns,_ = learnCodebooksAQ(winPoints, dim, M, K, \
                                                    winPoints.shape[0], None, 16, False, \
                                                    threadsCount, 10, \
                                                    codebooks, assigns[winIndex])
        
        codebooks[:] = winCodebooks
        assigns[winIndex] = winAssigns

        baseEndIndex = curBatchEndIndex

def onlineBatchAQ(baseFilename, dim, M, K, basePointsCount, numGroup, \
                  trainN, inferenceN, codebooksFilename, codeFilename, \
                  threadsCount=15, itsCount=20, \
                  k=1024,MODE='update'):
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
        print(startIndexPerBatch)
        for i in range(1, numGroup):
            print(startIndexPerBatch[i] - startIndexPerBatch[i-1])
        print(basePointsCount - startIndexPerBatch[numGroup-1])
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
        print(startIndexPerBatch)
        for i in range(1, numGroup):
            print(startIndexPerBatch[i] - startIndexPerBatch[i-1])
        print(basePointsCount - startIndexPerBatch[numGroup-1])
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
        print(startIndexPerBatch)
        for i in range(1, numGroup-1):
            print(startIndexPerBatch[i] - startIndexPerBatch[i-1])
        print(basePointsCount - startIndexPerBatch[numGroup-1])
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
                    trainN, inferenceN, codebooksFilename, codeFilename, \
                    threadsCount=threadsCount, itsCount=itsCount, \
                    k=k)
    elif MODE == 'rebatch':
        onlineReBatch(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                    trainN, inferenceN, codebooksFilename, codeFilename, \
                    threadsCount=threadsCount, itsCount=itsCount, \
                    k=k)
    elif MODE == 'noupdate':
        onlineBatchNoUpdateAQ(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                            trainN, inferenceN, codebooksFilename, codeFilename, \
                            threadsCount=threadsCount, itsCount=itsCount, \
                            k=k)

    elif MODE == 'retrain':
        onlineBatchRetrainAQ(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                            trainN, inferenceN, codebooksFilename, codeFilename, \
                            threadsCount=threadsCount, itsCount=itsCount, \
                            k=k)

    elif MODE == 'streaming':
        onlineStreamingAQ(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                            trainN, inferenceN, codebooksFilename, codeFilename, \
                            threadsCount=threadsCount, itsCount=itsCount, \
                            k=k)
    elif MODE == 'stream_batch':
        onlineStreamingBatchAQ(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                            trainN, inferenceN, codebooksFilename, codeFilename, \
                            threadsCount=threadsCount, itsCount=itsCount, \
                            k=k)
    
    elif MODE == 'win_streaming':
        onlineStreamingSlidingWindowAQ(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                                        trainN, inferenceN, codebooksFilename, codeFilename, \
                                        threadsCount=threadsCount, itsCount=itsCount, \
                                        k=k)

    elif MODE == 'win_batch':
        onlineBatchSlidingWindowAQ(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                                    trainN, inferenceN, codebooksFilename, codeFilename, \
                                    threadsCount=threadsCount, itsCount=itsCount, \
                                    k=k)

    elif MODE == 'win_retrain':
        onlineRetrainSlidingWindowAQ(points, startIndexPerBatch, dim, M, K, basePointsCount, numGroup, \
                                    trainN, inferenceN, codebooksFilename, codeFilename, \
                                    threadsCount=threadsCount, itsCount=itsCount, \
                                    k=k)
    
    elif MODE == 'osh':
        onlineBatchOSH(points, startIndexPerBatch, dim, 64, K, basePointsCount, numGroup, \
                        trainN, inferenceN, codebooksFilename, codeFilename, \
                        threadsCount=threadsCount, itsCount=itsCount, \
                        k=k)

    else:
        raise(f"Unexpected MODE {MODE}.")
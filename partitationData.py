import pickle
import random
import numpy as np
import pandas as pd
from scipy import io as matIO

from utils import *
from newIO import *
from methodParams import MethodParams
from dataParams import DataParams

from aqCoding import learnCodebooksAQ, encodePointsAQ

def partitation(dataset, param, numGroup=12):
    if param.M < 1:
        raise Exception("M is not positive.")
    
    dim = dataset.dim

    baseFilename = dataset.baseFilename
    basePointsCount = dataset.basePointsCount

    #caltech101 dataset  
    if baseFilename.endswith('caltech101.mat'):
        data = matIO.loadmat(baseFilename)
        rawdata = data['gist']
        points = np.zeros((basePointsCount, dim))
        startIndexPerBatch, points = genStartIndexPerBatch(rawdata, points, numGroup, classNum=102, is_shuffle=True)

    #halfdome dataset
    elif baseFilename.endswith('halfdome.mat'):
        data = matIO.loadmat(baseFilename)
        rawDataWithNan = data['data']
        df = pd.DataFrame(rawDataWithNan)
        newdf = df.dropna(axis=0, how='any')
        rawdata = newdf.values
        
        assert(rawdata.shape[0] == basePointsCount)

        points = np.zeros((basePointsCount, dim))
        startIndexPerBatch, points = genStartIndexPerBatchHalfDome(rawdata,'./data/halfdome/info.txt', points, numGroup, classNum=28086, is_shuffle=True)

    elif baseFilename.endswith('sun397.npy'):
        rawdata = np.load(baseFilename, allow_pickle=True)
        points = np.zeros((basePointsCount, dim))
        startIndexPerBatch, points = genStartIndexPerBatchSun397(rawdata, points, numGroup, classNum=397, is_shuffle=True)

    elif baseFilename.endswith('imagenet.npy'):
        rawdata = np.load(baseFilename, allow_pickle=True)
        points = np.zeros((basePointsCount, dim))
        startIndexPerBatch, points = genStartIndexPerBatchImageNet(rawdata, points, numGroup, classNum=1000, is_shuffle=True)

    else:
        raise('Read data error.')
    
    initPoints = points[:startIndexPerBatch[1]]

    np.save(dataset.startIndex, startIndexPerBatch)
    np.save(dataset.points, points)

    codebooks,assigns, oldAssigns = learnCodebooksAQ(initPoints, dataset.dim, param.M, param.K, \
                                                 startIndexPerBatch[1], dataset.codebooks, \
                                                 16, saveCodebook=False, \
                                                 threadsCount=10, \
                                                 itsCount=20)
    with open(dataset.codebooks, 'wb') as f:
        pickle.dump(codebooks, f)

    with open(dataset.initAssigns, 'wb') as f:
        pickle.dump(assigns, f)
    
    with open(dataset.initOldAssigns, 'wb') as f:
        pickle.dump(oldAssigns, f)

def ratioPartition(dataset, param, ratio=0.5, seed=None):


    if param.M < 1:
        raise Exception("M must be positive.")
    
    if seed is not None:
        np.random.seed(seed)

    dim = dataset.dim

    baseFilename = dataset.baseFilename
    basePointsCount = dataset.basePointsCount

    shuffled_indices=np.random.permutation(basePointsCount)
    trainCount = int(basePointsCount*ratio)

    print(f'All:{basePointsCount}, Train:{trainCount}, Test:{basePointsCount-trainCount}.')

    if 'caltech' in baseFilename:
        data = matIO.loadmat(baseFilename)
        rawdata = data['gist']
        numClass = rawdata.shape[0]
        points = np.vstack([rawdata[i][0] for i in range(numClass)])
        
    elif 'halfdome' in baseFilename:
        data = matIO.loadmat(baseFilename)
        rawDataWithNan = data['data']
        df = pd.DataFrame(rawDataWithNan)
        newdf = df.dropna(axis=0, how='any')
        points = newdf.values

        assert(points.shape[0] == basePointsCount)
    
    elif 'sun397' in baseFilename:
        # rawdata = np.load(baseFilename, allow_pickle=True)
        # numClass = rawdata.shape[0]
        # points = np.vstack([rawdata[i] for i in range(numClass)])
        data = matIO.loadmat('/amax/home/liuqi/gistdescriptor/sun397.mat')
        rawdata = data['gist']
        numClass = rawdata.shape[0]
        points = np.vstack([rawdata[i][0] for i in range(numClass)])
    
    elif 'imagenet' in baseFilename:
        rawdata = np.load(baseFilename, allow_pickle=True)
        numClass = rawdata.shape[0]
        points = np.vstack([rawdata[i] for i in range(numClass)])
    
    else:
        raise(f'ERROR:{baseFilename} does not exist.')

    trainPoints = points[shuffled_indices[:trainCount]]

    testPoints = points[shuffled_indices[trainCount:]]
    
    # np.save(dataset.trainPoints, trainPoints)
    # np.save(dataset.testPoints, testPoints)
    
    codebooks,assigns, oldAssigns = learnCodebooksAQ(trainPoints, dataset.dim, param.M, param.K, \
                                                    trainCount, dataset.codebooks, \
                                                    16, saveCodebook=False, \
                                                    threadsCount=8, \
                                                    itsCount=20)
    
    testAssigns, testErrors = encodePointsAQ(testPoints, codebooks, 64)

    print(f'Test data encode error: {np.mean(testErrors)}.')

    # with open(dataset.codebooks, 'wb') as f:
    #     pickle.dump(codebooks, f)

    # with open(dataset.initAssigns, 'wb') as f:
    #     pickle.dump(assigns, f)

    # with open(dataset.initOldAssigns, 'wb') as f:
    #     pickle.dump(oldAssigns, f)
    
    # with open(dataset.testAssigns, 'wb') as f:
    #     pickle.dump(testAssigns, f)

if __name__ == '__main__':
    #parameter setting
    param = MethodParams(M=8, K=256, train_N=16, inference_N=64)
    dataset = DataParams('caltech101')
    
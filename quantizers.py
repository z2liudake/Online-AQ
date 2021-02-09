from aqCoding import *
from pqCoding import *

# set folder for your quantization models structure
dataFolder = './loss/' 

class Quantizer():
    def __init__(self):
        self.prefix = 'base_'
    def trainCodebooks(self, data, params):
        pass
    def getQuantizationError(self, data, params):
        pass
    def encodeDataset(self, data, params):
        pass
    def searchNearestNeighbors(self, data, params, k=1024):
        pass
    def getCodebooksFilename(self, data, params):
        res = dataFolder + data.prefix + self.prefix + params.prefix + 'codebooks.dat'
        return res

    def getCodesFilename(self, data, params):    
        res = dataFolder + data.prefix + self.prefix + params.prefix + 'code.dat'
        return res

class PqQuantizer(Quantizer):
    def __init__(self, threadsCount, itCount):
        self.prefix = 'pq_'
        self.threadsCount = threadsCount
        self.itCount = itCount
    
    def trainCodebooks(self, data, params):
        codebooksFilename = self.getCodebooksFilename(data, params)
        print(f'Begin to train codebooks and codebookFilename is {codebooksFilename}.')
        learnCodebooksPQ(data.learnFilename, data.dim, \
                         params.M, params.K, \
                         data.learnPointsCount, codebooksFilename, \
                         self.threadsCount, self.itCount)
    
    def encodeDataset(self, data, params):
        codebooksFilename = self.getCodebooksFilename(data, params)
        codeFilename = self.getCodesFilename(data, params)
        print(f'Begin to encode dataset and codebookFilename is {codebooksFilename}, codeFilename is {codeFilename}.')
        encodeDatasetPQ(data.baseFilename, data.basePointsCount, \
                        codebooksFilename, codeFilename, \
                        self.threadsCount)
    
    def searchNearestNeighbors(self, data, params, k=10000):
        codebooksFilename = self.getCodebooksFilename(data, params)
        codeFilename = self.getCodesFilename(data, params)
        print(f'Begin to search nearest neighbor and codebookFilename is {codebooksFilename}, codeFilename is {codeFilename}.')
        return searchNearestNeighborsPQ(codeFilename, codebooksFilename, \
                                        data.queriesFilename, data.queriesCount, \
                                        k, self.threadsCount)
    
    def onlineBatch(self, data, params, numGroup, MODE='update'):
        codebooksFilename = self.getCodebooksFilename(data, params)
        codeFilename = self.getCodesFilename(data, params)
        print(f'Online PQ of batch mode.')
        onlineBatchPQ(data.baseFilename, data.dim, \
                      params.M, params.K, \
                      data.basePointsCount, numGroup, \
                      codebooksFilename, codeFilename, \
                      self.threadsCount, self.itCount,
                      k=1024, MODE=MODE)

class AqQuantizer(Quantizer):
    def __init__(self, threadsCount, itCount):
        self.prefix = 'aq_'
        self.threadsCount = threadsCount
        self.itCount = itCount

    def trainCodebooks(self, data, params):
        codebooksFilename = self.getCodebooksFilename(data, params)
        print(f'Begin to train codebooks and codebookFilename is {codebooksFilename}.')
        learnCodebooksAQ(data.learnFilename, data.dim, \
                         params.M, params.K, \
                         data.learnPointsCount, codebooksFilename, params.train_bs_N,\
                         saveCodebook=True, \
                         threadsCount=self.threadsCount,itsCount=self.itCount)
    
    def getQuantizationError(self, data, params):
        codebooksFilename = self.getCodebooksFilename(data, params)
        print(f'Begin to quantation and codebookFilename is {codebooksFilename}.')
        return getQuantizationErrorAQ(data.testFilename, data.dim, \
                                      data.testPointsCount, codebooksFilename, params.inference_bs_N)
    
    def encodeDataset(self, data, params):
        codebooksFilename = self.getCodebooksFilename(data, params)
        codeFilename = self.getCodesFilename(data, params)
        print(f'Begin to encode dataset and codebookFilename is {codebooksFilename}, codeFilename is {codeFilename}.')
        encodeDatasetAQ(data.baseFilename, data.basePointsCount, \
                        codebooksFilename, codeFilename, params.inference_bs_N)

    def searchNearestNeighbors(self, data, params, k=10000):
        codebooksFilename = self.getCodebooksFilename(data, params)
        codeFilename = self.getCodesFilename(data, params)
        print(f'Begin to search nearest neighbor and codebookFilename is {codebooksFilename}, codeFilename is {codeFilename}.')
        return searchNearestNeighborsAQ(codeFilename, codebooksFilename, \
                                        data.queriesFilename, data.queriesCount, \
                                        k, self.threadsCount)
    
    def onlineBatch(self, data, param, numGroup, MODE='update'):
        codebooksFilename = self.getCodebooksFilename(data, param)
        codesFilename = self.getCodesFilename(data, param)
        print(f'Online AQ of batch mode.')
        onlineBatchAQ(data.baseFilename, data.dim, \
                      param.M, param.K, \
                      data.basePointsCount, numGroup, \
                      param.train_bs_N, param.inference_bs_N, \
                      codebooksFilename, codesFilename, \
                      threadsCount=self.threadsCount, itsCount=self.itCount,
                      k=1024, MODE=MODE)
    
    def onlineBatchRL(self, data, param, numGroup):
        print('Online AQ of batch mode with RL Agent.')
        onlineBatchRLAQMain(data, param, numGroup, \
                            threadsCount=self.threadsCount, itsCount=self.itCount, \
                            k=1204)

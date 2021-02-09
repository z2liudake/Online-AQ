from quantizers import AqQuantizer,PqQuantizer
from methodParams import MethodParams
from dataParams import DataParams
from getRecall import getRecallAt
from newIO import ivecs_read

# parameters
## Set number of codebook M and codewards for each codebook K
paramSets = []
paramSets.append(MethodParams(M=4, K=256))

# datasets
## Set dataset to be used
datasets = []
# datasets.append(DataParams('siftsmall'))
datasets.append(DataParams('sift1M'))
# datasets.append(DataParams('gist1M'))

# methods
quantizers = []
# quantizers.append(AqQuantizer(threadsCount=20, itCount=20))
# quantizers.append(PqQuantizer(threadsCount=20, itCount=30))

trainCodebooks = True
encodeDatasets = False

if trainCodebooks:
    for params in paramSets:
        for data in datasets:
            for method in quantizers:
                method.trainCodebooks(data, params)
                print(f'Codebooks for settings {data.prefix}{method.prefix}{params.prefix} are learned')

if encodeDatasets:
    for params in paramSets:
        for data in datasets:
            for method in quantizers:
                method.encodeDataset(data, params)
                print(f'Dataset for settings {data.prefix}{method.prefix}{params.prefix}  is encoded')

# for params in paramSets:
#     for data in datasets:
#         for method in quantizers:
#             print (f'Settings: {data.prefix}{method.prefix}{params.prefix}')
#             print (f'Quantization error: {method.getQuantizationError(data, params)}')

# for params in paramSets:
#     for data in datasets:
#         for method in quantizers:
#             neighborsCount = min(1024, data.basePointsCount)
#             result = method.searchNearestNeighbors(data, params, k=neighborsCount)

#             groundtruth = ivecs_read(data.groundtruthFilename)
#             print (f'Results: {data.prefix}{method.prefix}{params.prefix}')
#             T = [2**i for i in range(11)]
#             T.insert(5, 20)
#             for i in T:
#                 print (f'Recall@{i} {getRecallAt(i, result, groundtruth)}')
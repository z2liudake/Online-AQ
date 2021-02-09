from quantizers import AqQuantizer, PqQuantizer
from methodParams import MethodParams
from dataParams import DataParams

#parameter setting
param = MethodParams(M=8, K=256, train_N=16, inference_N=64)

dataset = DataParams('caltech101')
# dataset = DataParams('halfdome')
# dataset = DataParams('sun397')
# dataset = DataParams('cifar10')

quantizer = AqQuantizer(threadsCount=30, itCount=60)
# quantizer = PqQuantizer(threadsCount=20, itCount=30)

numGroup = 12
MODE = 'rebatch'

quantizer.onlineBatch(dataset, param, numGroup, MODE=MODE)
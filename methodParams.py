class MethodParams:
    def __init__(self, M, K,train_N=16, inference_N=64):
        self.M = M # number of codebooks
        self.K = K # size of each codebook
        self.train_bs_N = train_N
        self.inference_bs_N = inference_N
        self.prefix = str(M) + '_' + str(K) + '_'

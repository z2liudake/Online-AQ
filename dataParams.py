class DataParams:
    def __init__(self, dataName):
        if dataName == 'sift1M':
            self.prefix = 'sift1M_'
            self.learnFilename = './data/sift/sift_learn.fvecs'
            self.testFilename = './data/sift/sift_query.fvecs'
            self.baseFilename = './data/sift/sift_base.fvecs'
            self.groundtruthFilename = './data/sift/sift_groundtruth.ivecs'
            self.queriesFilename = './data/sift/sift_query.fvecs'
            self.dim = 128
            self.learnPointsCount = 100000
            self.testPointsCount = 1000
            self.basePointsCount = 1000000
            self.queriesCount = 10000

        elif dataName == 'newsgroup20':
            self.prefix = 'newsgroup20_'
            self.baseFilename = './data/newsgroup20/newsgroup20.npy'
            self.dim = 300
            self.basePointsCount = 18846
            
        elif dataName == 'caltech101':
            self.prefix = 'caltech101_'
            self.baseFilename = './data/caltech101/caltech101.mat'
            self.startIndex = './Experiment/caltech101/startindex.npy'
            self.trainPoints = './Experiment/caltech101/trainPoints.npy'
            self.testPoints = './Experiment/caltech101/testPoints.npy'
            self.codebooks = './Experiment/caltech101/codebooks.mat'
            self.bestmodel = './Experiment/caltech101/best_model.pkl'
            self.initAssigns = './Experiment/caltech101/initassigns.mat'
            self.initOldAssigns = './Experiment/caltech101/initoldassigns.mat'
            self.testAssigns = './Experiment/caltech101/testassigns.mat'
            self.dim=512
            self.basePointsCount = 9145
            
        elif dataName == 'halfdome':
            self.prefix = 'halfdome_'
            self.baseFilename = './data/halfdome/halfdome.mat'
            self.startIndex = './Experiment/halfdome/startindex.npy'
            self.trainPoints = './Experiment/halfdome/trainPoints.npy'
            self.testPoints = './Experiment/halfdome/testPoints.npy'
            self.codebooks = './Experiment/halfdome/codebooks.mat'
            self.bestmodel = './Experiment/halfdome/best_model.pkl'
            self.initAssigns = './Experiment/halfdome/initassigns.mat'
            self.initOldAssigns = './Experiment/halfdome/initoldassigns.mat'
            self.testAssigns = './Experiment/halfdome/testassigns.mat'
            self.dim = 512
            self.basePointsCount = 107729
        
        elif dataName == 'sun397':
            self.prefix = 'sun397_'
            self.baseFilename = './data/sun397/sun397.mat'
            self.startIndex = './Experiment/sun397/startindex.npy'
            self.trainPoints = './Experiment/sun397/trainPoints.npy'
            self.testPoints = './Experiment/sun397/testPoints.npy'
            self.codebooks = './Experiment/sun397/codebooks.mat'
            self.bestmodel = './Experiment/sun397/best_model.pkl'
            self.initAssigns = './Experiment/sun397/initassigns.mat'
            self.initOldAssigns = './Experiment/sun397/initoldassigns.mat'
            self.testAssigns = './Experiment/sun397/testassigns.mat'
            self.dim = 512
            self.basePointsCount = 108754
        
        elif dataName == 'cifar10':
            self.prefix = 'cifar10_'
            self.baseFilename = './data/cifar10/cifar10.mat'
            self.startIndex = './Experiment/cifar10/startindex.npy'
            self.trainPoints = './Experiment/cifar10/trainPoints.npy'
            self.testPoints = './Experiment/cifar10/testPoints.npy'
            self.codebooks = './Experiment/cifar10/codebooks.mat'
            self.bestmodel = './Experiment/cifar10/best_model.pkl'
            self.initAssigns = './Experiment/cifar10/initassigns.mat'
            self.initOldAssigns = './Experiment/cifar10/initoldassigns.mat'
            self.testAssigns = './Experiment/cifar10/testassigns.mat'
            self.dim = 512
            self.basePointsCount = 60000
        
        elif dataName == 'imagenet':
            self.prefix = 'imagenet_'
            self.baseFilename = './data/imagenet/imagenet.npy'
            self.startIndex = './Experiment/imagenet/startindex.npy'
            self.trainPoints = './Experiment/imagenet/trainPoints.npy'
            self.testPoints = './Experiment/imagenet/testPoints.npy'
            self.codebooks = './Experiment/imagenet/codebooks.mat'
            self.bestmodel = './Experiment/imagenet/best_model.pkl'
            self.initAssigns = './Experiment/imagenet/initassigns.mat'
            self.initOldAssigns = './Experiment/imagenet/initoldassigns.mat'
            self.testAssigns = './Experiment/imagenet/testassigns.mat'
            self.dim = 512
            self.basePointsCount = 1331167

        elif dataName == 'siftsmall':
            self.prefix = 'siftsmall_'
            self.learnFilename = './data/siftsmall/siftsmall_learn.fvecs'
            self.testFilename = './data/siftsmall/siftsmall_query.fvecs'
            self.baseFilename = './data/siftsmall/siftsmall_base.fvecs'
            self.groundtruthFilename = './data/siftsmall/siftsmall_groundtruth.ivecs'
            self.queriesFilename = './data/siftsmall/siftsmall_query.fvecs'
            self.dim = 128
            self.learnPointsCount = 25000
            self.testPointsCount = 100
            self.basePointsCount = 10000
            self.queriesCount = 100
            
        elif dataName == 'gist1M':
            self.prefix = 'gist1M_'
            self.learnFilename = './data/gist/gist_learn.fvecs'
            self.testFilename = './data/gist/gist_query.fvecs'
            self.baseFilename = './data/gist/gist_base.fvecs'
            self.groundtruthFilename = './data/gist/gist_groundtruth.ivecs'
            self.queriesFilename = './data/gist/gist_query.fvecs'
            self.dim = 960
            self.learnPointsCount = 200000
            self.testPointsCount = 1000
            self.basePointsCount = 1000000
            self.queriesCount = 1000
            
        else:
            raise Exception("Unknown data!")


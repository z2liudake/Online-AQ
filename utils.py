import numpy as np
import random

def genStartIndexPerBatch(rawdata, points, numGroup, classNum=102, is_shuffle=False):
    classPerBatch = np.ones(numGroup, dtype='int32')*(rawdata.shape[0]//numGroup)

    startIndexPerBatch = np.zeros(numGroup, dtype="int32")
    leftClassNum = rawdata.shape[0] - numGroup*(rawdata.shape[0]//numGroup)

    for i in range(leftClassNum):
        classPerBatch[numGroup-1-i] += 1

    cumClassBatch = np.cumsum(classPerBatch)
   
    if is_shuffle:
        random.seed(8)
        shuffleID = list(range(classNum))
        random.shuffle(shuffleID)

    fisrstStartIndex = 0
    firstEndIndex = 0
    secondStartIndex = 0
    secondEndIndex = 0

    for i in range(numGroup//2):
        firstBatchClassNum = classPerBatch[2*i]
        secondBatchClassNum = classPerBatch[2*i+1]

        firstBatchTotalImage = 0
        totalImage = 0

        for j in  range(firstBatchClassNum):
            classID = 0 + j if i==0 else cumClassBatch[2*i-1]+j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]][0]
            else:
                curData = rawdata[classID][0]
            
            firstBatchTotalImage += curData.shape[0]//2
            totalImage += curData.shape[0]

        for j in range(secondBatchClassNum):
            classID = cumClassBatch[2*i] + j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]][0]
            else:
                curData = rawdata[classID][0]
            firstBatchTotalImage += curData.shape[0]//2
            totalImage += curData.shape[0]
        
        fisrstStartIndex = secondEndIndex
        secondStartIndex = secondEndIndex + firstBatchTotalImage

        startIndexPerBatch[2*i] = fisrstStartIndex
        startIndexPerBatch[2*i+1] = secondStartIndex

        for j in range(firstBatchClassNum):
            classID = 0 + j if i==0 else cumClassBatch[2*i-1]+j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]][0]
            else:
                curData = rawdata[classID][0]
            
            n = curData.shape[0]
            firstEndIndex = fisrstStartIndex + n//2
            secondEndIndex = secondStartIndex +(n-n//2)

            points[fisrstStartIndex:firstEndIndex,:] = curData[:n//2,:]
            points[secondStartIndex:secondEndIndex,:] = curData[n//2:,:]

            fisrstStartIndex = firstEndIndex
            secondStartIndex = secondEndIndex
        
        for j in range(secondBatchClassNum):
            classID = cumClassBatch[2*i]+j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]][0]
            else:
                curData = rawdata[classID][0]
            
            n = curData.shape[0]
            firstEndIndex = fisrstStartIndex + n//2
            secondEndIndex = secondStartIndex +(n-n//2)

            points[fisrstStartIndex:firstEndIndex,:] = curData[:n//2,:]
            points[secondStartIndex:secondEndIndex,:] = curData[n//2:,:]

            fisrstStartIndex = firstEndIndex
            secondStartIndex = secondEndIndex
        
    return startIndexPerBatch, points

def genStartIndexPerBatchSW(rawdata, points, numGroup, classNum=102, is_shuffle=False):
    classPerBatch = np.ones(numGroup, dtype='int32')*(rawdata.shape[0]//numGroup)

    startIndexPerBatch = np.zeros(numGroup, dtype="int32")
    leftClassNum = rawdata.shape[0] - numGroup*(rawdata.shape[0]//numGroup)

    # for i in range(leftClassNum):
    classPerBatch[0] += leftClassNum
    
    cumClassBatch = np.cumsum(classPerBatch)
    
    if is_shuffle:
        random.seed(67)
        shuffleID = list(range(classNum))
        random.shuffle(shuffleID)
    
    startIndex = 0
    endIndex = 0
    for i in range(numGroup):

        for j in range(classPerBatch[i]):
            
            if i==0:
                offset = 0
            else:
                offset = cumClassBatch[i-1]

            if is_shuffle:
                classID = shuffleID[j + offset]
            else:
                classID = j + offset

            curData = rawdata[classID][0]
            
            endIndex += curData.shape[0]

            points[startIndex:endIndex,:] = curData

            startIndex = endIndex

        if i != numGroup-1:
            startIndexPerBatch[i+1] = endIndex

    return startIndexPerBatch, points

def genStartIndexPerBatchHalfDome(data, pathToInfo, points, numGroup, classNum=28086, is_shuffle=False):
    with open(pathToInfo, mode='r') as f:
        rawlines = f.readlines()
    
    lines = []
    for idx, line in enumerate(rawlines):
        if idx not in [90504, 90507, 90520]:
            lines.append(line)

    rawdata = []
    
    startIndex = 0
    endIndex = 0
    curClassId = 1
    
    lenOfLines = len(lines)

    for idx,line in enumerate(lines):
        classID,_ = line.strip().split()
        classID = int(classID)

        endIndex = idx

        if curClassId != classID:
            classData = np.zeros((endIndex-startIndex,512))
            classData[:] = data[startIndex:endIndex,:]
            rawdata.append(classData)
            curClassId = classID
            startIndex = endIndex
        
        elif idx == lenOfLines-1:
            classData = np.zeros((endIndex+1-startIndex,512))
            classData[:] = data[startIndex:endIndex+1,:]
            rawdata.append(classData)
            
    #begin to generate startIndexPerBatch
    classPerBatch = np.ones(numGroup, dtype='int32')*(len(rawdata)//numGroup)

    startIndexPerBatch = np.zeros(numGroup, dtype="int32")
    leftClassNum = len(rawdata) - numGroup*(len(rawdata)//numGroup)

    for i in range(leftClassNum):
        classPerBatch[numGroup-1-i] += 1

    cumClassBatch = np.cumsum(classPerBatch)
   
    if is_shuffle:
        random.seed(3)
        shuffleID = list(range(classNum))
        random.shuffle(shuffleID)

    fisrstStartIndex = 0
    firstEndIndex = 0
    secondStartIndex = 0
    secondEndIndex = 0

    for i in range(numGroup//2):
        firstBatchClassNum = classPerBatch[2*i]
        secondBatchClassNum = classPerBatch[2*i+1]

        firstBatchTotalImage = 0

        for j in  range(firstBatchClassNum):
            classID = 0 + j if i==0 else cumClassBatch[2*i-1]+j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]]
            else:
                curData = rawdata[classID]
            
            firstBatchTotalImage += curData.shape[0]//2
 
        for j in range(secondBatchClassNum):
            classID = cumClassBatch[2*i] + j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]]
            else:
                curData = rawdata[classID]
            firstBatchTotalImage += curData.shape[0]//2

        
        fisrstStartIndex = secondEndIndex
        secondStartIndex = secondEndIndex + firstBatchTotalImage

        startIndexPerBatch[2*i] = fisrstStartIndex
        startIndexPerBatch[2*i+1] = secondStartIndex

        for j in range(firstBatchClassNum):
            classID = 0 + j if i==0 else cumClassBatch[2*i-1]+j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]]
            else:
                curData = rawdata[classID]
            
            n = curData.shape[0]
            firstEndIndex = fisrstStartIndex + n//2
            secondEndIndex = secondStartIndex +(n-n//2)

            points[fisrstStartIndex:firstEndIndex,:] = curData[:n//2,:]
            points[secondStartIndex:secondEndIndex,:] = curData[n//2:,:]

            fisrstStartIndex = firstEndIndex
            secondStartIndex = secondEndIndex
        
        for j in range(secondBatchClassNum):
            classID = cumClassBatch[2*i]+j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]]
            else:
                curData = rawdata[classID]
            
            n = curData.shape[0]
            firstEndIndex = fisrstStartIndex + n//2
            secondEndIndex = secondStartIndex +(n-n//2)

            points[fisrstStartIndex:firstEndIndex,:] = curData[:n//2,:]
            points[secondStartIndex:secondEndIndex,:] = curData[n//2:,:]

            fisrstStartIndex = firstEndIndex
            secondStartIndex = secondEndIndex
        
    return startIndexPerBatch, points

def genStartIndexPerBatchHalfDomeSW(data, pathToInfo, points, numGroup, classNum=28086, is_shuffle=False):
    with open(pathToInfo, mode='r') as f:
        rawlines = f.readlines()
    
    lines = []
    for idx, line in enumerate(rawlines):
        if idx not in [90504, 90507, 90520]:
            lines.append(line)

    rawdata = []
    
    startIndex = 0
    endIndex = 0
    curClassId = 1
    
    lenOfLines = len(lines)

    for idx,line in enumerate(lines):
        classID,_ = line.strip().split()
        classID = int(classID)

        endIndex = idx

        if curClassId != classID:
            classData = np.zeros((endIndex-startIndex,512))
            classData[:] = data[startIndex:endIndex,:]
            rawdata.append(classData)
            curClassId = classID
            startIndex = endIndex
        
        elif idx == lenOfLines-1:
            classData = np.zeros((endIndex+1-startIndex,512))
            classData[:] = data[startIndex:endIndex+1,:]
            rawdata.append(classData)
    startIndexPerBatch = np.zeros(numGroup, dtype="int32")

    #begin to generate startIndexPerBatch
    # classPerBatch = np.ones(numGroup, dtype='int32')*(len(rawdata)//numGroup)
    # leftClassNum = len(rawdata) - numGroup*(len(rawdata)//numGroup)

    # classPerBatch[0] += leftClassNum

    #1/3 dataset for the initial data
    classPerBatch = np.ones(numGroup, dtype='int32')
    classPerBatch[0] = len(rawdata)//3
    leftClassNum = len(rawdata) - len(rawdata)//3
    classPerBatch[1:] = leftClassNum//(numGroup-1)
    classPerBatch[1] += (leftClassNum - (numGroup-1)*(leftClassNum//(numGroup-1)))

    cumClassBatch = np.cumsum(classPerBatch)
   
    if is_shuffle:
        random.seed(1)
        shuffleID = list(range(classNum))
        random.shuffle(shuffleID)
    
    startIndex = 0
    endIndex = 0
    for i in range(numGroup):

        for j in range(classPerBatch[i]):
            
            if i==0:
                offset = 0
            else:
                offset = cumClassBatch[i-1]

            if is_shuffle:
                classID = shuffleID[j + offset]
            else:
                classID = j + offset

            curData = rawdata[classID]
            
            endIndex += curData.shape[0]

            points[startIndex:endIndex,:] = curData

            startIndex = endIndex

        if i != numGroup-1:
            startIndexPerBatch[i+1] = endIndex

    return startIndexPerBatch, points

def genStartIndexPerBatchSun397(rawdata, points, numGroup, classNum=397, is_shuffle=False):
    classPerBatch = np.ones(numGroup, dtype='int32')*(397//numGroup)

    startIndexPerBatch = np.zeros(numGroup, dtype="int32")
    leftClassNum = 397 - numGroup*(397//numGroup)

    for i in range(leftClassNum):
        classPerBatch[numGroup-1-i] += 1

    cumClassBatch = np.cumsum(classPerBatch)
   
    if is_shuffle:
        random.seed(189)
        shuffleID = list(range(classNum))
        random.shuffle(shuffleID)

    fisrstStartIndex = 0
    firstEndIndex = 0
    secondStartIndex = 0
    secondEndIndex = 0

    for i in range(numGroup//2):
        firstBatchClassNum = classPerBatch[2*i]
        secondBatchClassNum = classPerBatch[2*i+1]

        firstBatchTotalImage = 0
        totalImage = 0

        for j in  range(firstBatchClassNum):
            classID = 0 + j if i==0 else cumClassBatch[2*i-1]+j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]][0]
            else:
                curData = rawdata[classID][0]
            
            firstBatchTotalImage += curData.shape[0]//2
            totalImage += curData.shape[0]

        for j in range(secondBatchClassNum):
            classID = cumClassBatch[2*i] + j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]][0]
            else:
                curData = rawdata[classID][0]
            firstBatchTotalImage += curData.shape[0]//2
            totalImage += curData.shape[0]
        
        fisrstStartIndex = secondEndIndex
        secondStartIndex = secondEndIndex + firstBatchTotalImage

        startIndexPerBatch[2*i] = fisrstStartIndex
        startIndexPerBatch[2*i+1] = secondStartIndex

        for j in range(firstBatchClassNum):
            classID = 0 + j if i==0 else cumClassBatch[2*i-1]+j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]][0]
            else:
                curData = rawdata[classID][0]
            
            n = curData.shape[0]
            firstEndIndex = fisrstStartIndex + n//2
            secondEndIndex = secondStartIndex +(n-n//2)

            points[fisrstStartIndex:firstEndIndex,:] = curData[:n//2,:]
            points[secondStartIndex:secondEndIndex,:] = curData[n//2:,:]

            fisrstStartIndex = firstEndIndex
            secondStartIndex = secondEndIndex
        
        for j in range(secondBatchClassNum):
            classID = cumClassBatch[2*i]+j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]][0]
            else:
                curData = rawdata[classID][0]
            
            n = curData.shape[0]
            firstEndIndex = fisrstStartIndex + n//2
            secondEndIndex = secondStartIndex +(n-n//2)

            points[fisrstStartIndex:firstEndIndex,:] = curData[:n//2,:]
            points[secondStartIndex:secondEndIndex,:] = curData[n//2:,:]

            fisrstStartIndex = firstEndIndex
            secondStartIndex = secondEndIndex
    
    if numGroup%2 == 1:
        startIndexPerBatch[numGroup-1]=secondStartIndex
        for j in range(classPerBatch[numGroup-1]):
            classID = cumClassBatch[numGroup-2]+j
            if is_shuffle:
                curData = rawdata[shuffleID[classID]][0]
            else:
                curData = rawdata[classID][0]
            n = curData.shape[0]
            points[secondStartIndex:secondStartIndex+n] = curData
            secondStartIndex += n

    return startIndexPerBatch, points

def genStartIndexPerBatchSun397SW(rawdata, points, numGroup, classNum=397, is_shuffle=False):
    startIndexPerBatch = np.zeros(numGroup, dtype="int32")
    # classPerBatch = np.ones(numGroup, dtype='int32')*(397//numGroup)

    # startIndexPerBatch = np.zeros(numGroup, dtype="int32")
    # leftClassNum = 397 - numGroup*(397//numGroup)

    # classPerBatch[0] += leftClassNum

    #1/3 data for initial data
    classPerBatch = np.ones(numGroup, dtype='int32')
    classPerBatch[0] = len(rawdata)//3
    leftClassNum = len(rawdata) - len(rawdata)//3
    classPerBatch[1:] = leftClassNum//(numGroup-1)
    classPerBatch[1] += (leftClassNum - (numGroup-1)*(leftClassNum//(numGroup-1)))

    cumClassBatch = np.cumsum(classPerBatch)
   
    if is_shuffle:
        random.seed(2)
        shuffleID = list(range(classNum))
        random.shuffle(shuffleID)
    
    startIndex = 0
    endIndex = 0
    for i in range(numGroup):

        for j in range(classPerBatch[i]):
            
            if i==0:
                offset = 0
            else:
                offset = cumClassBatch[i-1]

            if is_shuffle:
                classID = shuffleID[j + offset]
            else:
                classID = j + offset

            curData = rawdata[classID][0]
            
            endIndex += curData.shape[0]

            points[startIndex:endIndex,:] = curData

            startIndex = endIndex

        if i != numGroup-1:
            startIndexPerBatch[i+1] = endIndex

    return startIndexPerBatch, points

def genStartIndexPerBatchCifar10(rawdata, points, numGroup, classNum=10, is_shuffle=False):
    classPerBatch = np.ones(numGroup, dtype='int32')*(rawdata.shape[0]//numGroup)

    startIndexPerBatch = np.zeros(numGroup, dtype="int32")
    leftClassNum = rawdata.shape[0] - numGroup*(rawdata.shape[0]//numGroup)

    for i in range(leftClassNum):
        classPerBatch[numGroup-1-i] += 1

    cumClassBatch = np.cumsum(classPerBatch)
   
    if is_shuffle:
        random.seed(3)
        shuffleID = list(range(classNum))
        random.shuffle(shuffleID)

    fisrstStartIndex = 0
    firstEndIndex = 0
    secondStartIndex = 0
    secondEndIndex = 0

    for i in range(numGroup//2):
        firstBatchClassNum = classPerBatch[2*i]
        secondBatchClassNum = classPerBatch[2*i+1]

        firstBatchTotalImage = 0
        totalImage = 0

        for j in  range(firstBatchClassNum):
            classID = 0 + j if i==0 else cumClassBatch[2*i-1]+j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]][0]
            else:
                curData = rawdata[classID][0]
            
            firstBatchTotalImage += curData.shape[0]//2
            totalImage += curData.shape[0]

        for j in range(secondBatchClassNum):
            classID = cumClassBatch[2*i] + j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]][0]
            else:
                curData = rawdata[classID][0]
            firstBatchTotalImage += curData.shape[0]//2
            totalImage += curData.shape[0]
        
        fisrstStartIndex = secondEndIndex
        secondStartIndex = secondEndIndex + firstBatchTotalImage

        startIndexPerBatch[2*i] = fisrstStartIndex
        startIndexPerBatch[2*i+1] = secondStartIndex

        for j in range(firstBatchClassNum):
            classID = 0 + j if i==0 else cumClassBatch[2*i-1]+j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]][0]
            else:
                curData = rawdata[classID][0]
            
            n = curData.shape[0]
            firstEndIndex = fisrstStartIndex + n//2
            secondEndIndex = secondStartIndex +(n-n//2)

            points[fisrstStartIndex:firstEndIndex,:] = curData[:n//2,:]
            points[secondStartIndex:secondEndIndex,:] = curData[n//2:,:]

            fisrstStartIndex = firstEndIndex
            secondStartIndex = secondEndIndex
        
        for j in range(secondBatchClassNum):
            classID = cumClassBatch[2*i]+j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]][0]
            else:
                curData = rawdata[classID][0]
            
            n = curData.shape[0]
            firstEndIndex = fisrstStartIndex + n//2
            secondEndIndex = secondStartIndex +(n-n//2)

            points[fisrstStartIndex:firstEndIndex,:] = curData[:n//2,:]
            points[secondStartIndex:secondEndIndex,:] = curData[n//2:,:]

            fisrstStartIndex = firstEndIndex
            secondStartIndex = secondEndIndex
        
    return startIndexPerBatch, points

def genStartIndexPerBatchCifar10SW(rawdata, points, numGroup, classNum=10, is_shuffle=False):
    startIndexPerBatch = np.zeros(numGroup, dtype="int32")

    # classPerBatch = np.ones(numGroup, dtype='int32')*(rawdata.shape[0]//numGroup)
    # leftClassNum = rawdata.shape[0] - numGroup*(rawdata.shape[0]//numGroup)
    # classPerBatch[0] += leftClassNum

    classPerBatch = np.ones(numGroup, dtype='int32')
    classPerBatch[0] += 1
    
    cumClassBatch = np.cumsum(classPerBatch)
    
    if is_shuffle:
        random.seed(1)
        shuffleID = list(range(classNum))
        random.shuffle(shuffleID)
    
    startIndex = 0
    endIndex = 0
    for i in range(numGroup):

        for j in range(classPerBatch[i]):
            
            if i==0:
                offset = 0
            else:
                offset = cumClassBatch[i-1]

            if is_shuffle:
                classID = shuffleID[j + offset]
            else:
                classID = j + offset

            curData = rawdata[classID][0]
            
            endIndex += curData.shape[0]

            points[startIndex:endIndex,:] = curData

            startIndex = endIndex

        if i != numGroup-1:
            startIndexPerBatch[i+1] = endIndex

    return startIndexPerBatch, points

def genStartIndexPerBatchImageNet(rawdata, points, numGroup, classNum=1000, is_shuffle=False):
    classPerBatch = np.ones(numGroup, dtype='int32')*(1000//numGroup)

    startIndexPerBatch = np.zeros(numGroup, dtype="int32")
    leftClassNum = 1000 - numGroup*(1000//numGroup)

    for i in range(leftClassNum):
        classPerBatch[numGroup-1-i] += 1

    cumClassBatch = np.cumsum(classPerBatch)
   
    if is_shuffle:
        shuffleID = list(range(classNum))
        random.shuffle(shuffleID)

    fisrstStartIndex = 0
    firstEndIndex = 0
    secondStartIndex = 0
    secondEndIndex = 0

    for i in range(numGroup//2):
        firstBatchClassNum = classPerBatch[2*i]
        secondBatchClassNum = classPerBatch[2*i+1]

        firstBatchTotalImage = 0
        totalImage = 0

        for j in  range(firstBatchClassNum):
            classID = 0 + j if i==0 else cumClassBatch[2*i-1]+j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]]
            else:
                curData = rawdata[classID]
            
            firstBatchTotalImage += curData.shape[0]//2
            totalImage += curData.shape[0]

        for j in range(secondBatchClassNum):
            classID = cumClassBatch[2*i] + j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]]
            else:
                curData = rawdata[classID]
            firstBatchTotalImage += curData.shape[0]//2
            totalImage += curData.shape[0]
        
        fisrstStartIndex = secondEndIndex
        secondStartIndex = secondEndIndex + firstBatchTotalImage

        startIndexPerBatch[2*i] = fisrstStartIndex
        startIndexPerBatch[2*i+1] = secondStartIndex

        for j in range(firstBatchClassNum):
            classID = 0 + j if i==0 else cumClassBatch[2*i-1]+j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]]
            else:
                curData = rawdata[classID]
            
            n = curData.shape[0]
            firstEndIndex = fisrstStartIndex + n//2
            secondEndIndex = secondStartIndex +(n-n//2)

            points[fisrstStartIndex:firstEndIndex,:] = curData[:n//2,:]
            points[secondStartIndex:secondEndIndex,:] = curData[n//2:,:]

            fisrstStartIndex = firstEndIndex
            secondStartIndex = secondEndIndex
        
        for j in range(secondBatchClassNum):
            classID = cumClassBatch[2*i]+j
            
            if is_shuffle:
                curData = rawdata[shuffleID[classID]]
            else:
                curData = rawdata[classID]
            
            n = curData.shape[0]
            firstEndIndex = fisrstStartIndex + n//2
            secondEndIndex = secondStartIndex +(n-n//2)

            points[fisrstStartIndex:firstEndIndex,:] = curData[:n//2,:]
            points[secondStartIndex:secondEndIndex,:] = curData[n//2:,:]

            fisrstStartIndex = firstEndIndex
            secondStartIndex = secondEndIndex
         
    return startIndexPerBatch, points
import numpy as np
np.set_printoptions(threshold=np.nan)
nrOfParameters = 133



def loadDataFromFile (firstFileName, secondFileName):
    nrOfRows = 0
    with open(firstFileName, 'r') as f:
        for i, x in enumerate(f):
            if i == 1:
                nrOfRows = int(x)
                break

    firstImage = np.loadtxt(firstFileName, delimiter=' ', dtype=int, skiprows=2).reshape(nrOfRows, nrOfParameters)
    with open(secondFileName, 'r') as f:
        for i, x in enumerate(f):
            if i == 1:
                nrOfRows = int(x)
                break

    secondImage = np.loadtxt(secondFileName, delimiter=' ', dtype=int, skiprows=2).reshape(nrOfRows,
                                                                                                   nrOfParameters)
    return firstImage, secondImage

def countDifference (firstImage, secondImage):
    firstImageSize = firstImage.shape[0]
    secondImageSize = secondImage.shape[0]
    keyPoints = np.zeros((firstImageSize, 5), dtype=int)
    for i in range (0,firstImageSize):
        firstImageTraits = firstImage[i][5:]
        similarityValues = []
        similarityCoordinates = []
        for j in range (0,secondImageSize):
            secondImageTraits = secondImage[j][5:]
            similarityValue = np.sum(np.abs(firstImageTraits - secondImageTraits))
            similarityValues.append(similarityValue)
            similarityCoordinates.append(secondImage[j][0:2])

        mostSimilarPoint = np.argmin(similarityValues)
        keyPoint = np.array([firstImage[i][0:2], similarityCoordinates[mostSimilarPoint]]).flatten().tolist()
        keyPoint.append(similarityValues[mostSimilarPoint])
        keyPoints[i] = keyPoint
    keyPoints1 = np.zeros((secondImageSize, 5), dtype=int)
    for i in range(0, secondImageSize):
        secondImageTraits = secondImage[i][5:]
        similarityValues = []
        similarityCoordinates = []
        for j in range(0, firstImageSize):
            firstImageTraits = firstImage[j][5:]
            similarityValue = np.sum(np.abs(secondImageTraits - firstImageTraits))
            similarityValues.append(similarityValue)
            similarityCoordinates.append(firstImage[j][0:2])

        mostSimilarPoint = np.argmin(similarityValues)
        keyPoint = np.array([secondImage[i][0:2], similarityCoordinates[mostSimilarPoint]]).flatten().tolist()
        keyPoint.append(similarityValues[mostSimilarPoint])
        keyPoints1[i] = keyPoint
    return keyPoints, keyPoints1

def countKeyPointsPairs (firstImageKeyPoints, secondImageKeyPoints):
    firstImageSize = firstImageKeyPoints.shape[0]
    secondImageSize = secondImageKeyPoints.shape[0]
    keyPointsPairs = []
    for i in range (0, firstImageSize):
        for j in range (0, secondImageSize):
            if np.all(firstImageKeyPoints[i][0:2] == secondImageKeyPoints[j][2:4]):
                if np.all(firstImageKeyPoints[i][2:4] == secondImageKeyPoints[j][0:2]):
                    keyPointsPairs.append(np.array(firstImageKeyPoints[i][0:4]))
                    break
    return np.array(keyPointsPairs)
firstImage, secondImage = loadDataFromFile('img31.png.haraff.sift', 'img32.png.haraff.sift')

firstImageKeyPoints, secondImageKeyPoints = countDifference(firstImage, secondImage)

print(countKeyPointsPairs(firstImageKeyPoints, secondImageKeyPoints))
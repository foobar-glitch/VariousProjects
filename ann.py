import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
"""

def relu(x):
    return np.maximum(0, x)

def relu_grad(z):
    grad = np.zeros_like(z)
    grad[z > 0] = 1
    return grad

def softmax(x):
    x = np.asarray(x)

    # Case 1: column vector (n, 1) → treat as single sample with n classes
    if x.ndim == 2 and x.shape[1] == 1:
        x = x.T  # (1, n)

    # Case 2: row vector (1, n) or batch (batch, n)
    x = x - np.max(x, axis=-1, keepdims=True)
    expValues = np.exp(x)
    probs = expValues / np.sum(expValues, axis=-1, keepdims=True)

    # Return original shape if input was (n, 1)
    if probs.shape[0] == 1 and probs.shape[1] > 1:
        return probs.T

    return probs

def crossEntropy(yCorrect, yCalc):
    yVec = np.multiply(yCorrect, np.log(yCalc))
    return -np.sum(yVec)


def processImage(imagePath):
    img = Image.open(imagePath)
    img = img.convert('L')
    img = img.resize((28,28))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.flatten()
    return img_array


def calcOutput():
    pass

inputLayerSize = 784
hiddenLayerSize = 500
outputLayerSize = 10

numberOfHiddenLayers = 5

inputLayerWeights = np.random.randn(hiddenLayerSize, inputLayerSize) * np.sqrt(2./inputLayerSize)
inputLayerBias = np.random.randn(hiddenLayerSize, 1) * np.sqrt(2./1)

outputLayerWeights = np.random.randn(outputLayerSize, hiddenLayerSize) * np.sqrt(2./hiddenLayerSize)
outputLayerBias = np.random.randn(outputLayerSize, 1) * np.sqrt(2./1)


hiddenLayerWeights = np.random.randn(numberOfHiddenLayers, hiddenLayerSize, hiddenLayerSize) * np.sqrt(2./hiddenLayerSize)
hiddenLayerBias = np.random.randn(numberOfHiddenLayers, hiddenLayerSize, 1) * np.sqrt(2./hiddenLayerSize)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

trainingSetSize = y_train.size-1
learningRate = 0.01

for j in range(trainingSetSize):
    correctProb = np.zeros((10,1))
    correctProb[y_train[j]]=1

    hiddenLayerA = np.zeros((numberOfHiddenLayers, hiddenLayerSize, 1), dtype=float)
    hiddenLayerZ = np.zeros((numberOfHiddenLayers, hiddenLayerSize, 1), dtype=float)

    # x_picture: singular picture
    xPictureVector = x_train[j].flatten().reshape(-1, 1) / 255.0
    zInputLayer =  np.add(inputLayerWeights @ xPictureVector, inputLayerBias)
    aInputLayer = relu(zInputLayer)
    
    hiddenLayerZ[0] = np.add(hiddenLayerWeights[0] @ aInputLayer, hiddenLayerBias[0])
    hiddenLayerA[0] = relu(hiddenLayerZ[0])

    for i in range(1, numberOfHiddenLayers):
        hiddenLayerZ[i] = np.add(hiddenLayerWeights[i] @ hiddenLayerA[i-1], hiddenLayerBias[i])
        hiddenLayerA[i] = relu(hiddenLayerZ[i])
    
    zOutputLayer = np.add(outputLayerWeights @ hiddenLayerA[numberOfHiddenLayers-1], outputLayerBias)
    outProbs = softmax(zOutputLayer)

    print(y_train[j])
    print(outProbs.round(4).tolist())
    print("=====")

    #gradOutputBias = gammaOutput
    gammaOutput = outProbs-correctProb
    gradOutputWeight = gammaOutput @ hiddenLayerA[numberOfHiddenLayers-1].T


    gammaHidden = np.zeros((numberOfHiddenLayers, hiddenLayerSize, 1), dtype=float)
    gradHiddenWeight = np.zeros((numberOfHiddenLayers, hiddenLayerSize, hiddenLayerSize), dtype=float)


    gammaHidden[numberOfHiddenLayers-1] = (outputLayerWeights.T @ gammaOutput) * relu_grad(hiddenLayerZ[numberOfHiddenLayers-1])
    gradHiddenWeight[numberOfHiddenLayers-1] = gammaHidden[numberOfHiddenLayers-1] @ hiddenLayerA[numberOfHiddenLayers-2].T

    for k in range(numberOfHiddenLayers-2, -1, -1):
        gammaHidden[k] = (hiddenLayerWeights[k+1].T @ gammaHidden[k+1]) * relu_grad(hiddenLayerZ[k])
        if k == 0: break
        gradHiddenWeight[k] = gammaHidden[k] @ hiddenLayerA[k-1].T
    
    gradHiddenWeight[0] = gammaHidden[0] @ aInputLayer.T
    
    gammaInputLayer = (hiddenLayerWeights[0].T @ gammaHidden[0]) * relu_grad(zInputLayer)
    gradInputLayerWeight = gammaInputLayer @ xPictureVector.T


    inputLayerWeights = inputLayerWeights - learningRate*gradInputLayerWeight
    inputLayerBias = inputLayerBias - learningRate*gammaInputLayer

    hiddenLayerWeights = hiddenLayerWeights - learningRate*gradHiddenWeight
    hiddenLayerBias = hiddenLayerBias - learningRate*gammaHidden

    outputLayerWeights = outputLayerWeights - learningRate*gradOutputWeight
    outputLayerBias = outputLayerBias - learningRate*gammaOutput
    





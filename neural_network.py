import numpy as np

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def sigmoid_derivative(p):
    return p * (1 - p)

class NeuralNetwork:

    #Do not change this function header
    def __init__(self,x=[[]],y=[],numLayers=2,numNodes=2,eta=0.001,maxIter=10000):
        self.data = np.append(x,np.ones([len(x),1]),1)
        self.labels = np.array(y)
        self.nLayers = numLayers
        self.nNodes = numNodes
        self.eta = eta
        self.maxIt = maxIter
        self.weights = list()
        self.outputs = list()
        self.weights.append(np.random.rand(len(x[0])+1,self.nNodes))
        for index in range(self.nLayers-1):
            self.weights.append(np.random.rand(self.nNodes+1,self.nNodes))
        self.weights.append(np.random.rand(self.nNodes+1,1))

        for index in range(int(self.maxIt/90)):
            self.train(self.data)


    def train(self,x=[[]]):
        for index in range(len(x)):
            self.feedforward(self.data[index])
            self.backprop(self.data[index],self.labels[index])

    def predict(self,x=[]):
        self.feedforward(np.append(x,1))
        return self.outputs.pop()[0]

    def feedforward(self,point):
        self.outputs = list()
        self.outputs.append(np.append(sigmoid(np.dot(point,self.weights[0])),1))
        for index in range(1,len(self.weights)-1):
            self.outputs.append(np.append(sigmoid(np.dot(self.outputs[index-1],self.weights[index])),1))
        self.outputs.append(sigmoid(np.dot(self.outputs[len(self.outputs)-1],self.weights[len(self.weights)-1])))

    def backprop(self, point, lable):
        sensitivity=[]
        copyOutputs=self.outputs.copy()
        output=np.array(copyOutputs.pop())
        sensitivity.append((lable-output)*sigmoid_derivative(output))
        while len(copyOutputs)!=0:
            sensitivity.append(np.multiply(np.dot(sensitivity[len(sensitivity)-1],self.weights[len(copyOutputs)].T),sigmoid_derivative(copyOutputs.pop()))[:-1])
        sensitivity.reverse()
        changeWeight=[]
        changeWeight.append(np.array([np.multiply(np.multiply(self.outputs[len(sensitivity)-2],sensitivity[len(sensitivity)-1]),self.eta)]).T)
        for index in range(len(sensitivity)-2,0,-1):
            changeWeight.append(np.multiply(np.outer(self.outputs[index-1],sensitivity[index]),self.eta))
        changeWeight.append(np.multiply(np.outer(point,sensitivity[0]),self.eta))

        # print(self.weights)
        for index in range(len(self.weights)):
            self.weights[index]+=(changeWeight[len(changeWeight)-index-1])
        # print(self.weights)

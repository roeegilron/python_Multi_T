import numpy as np

from calcMultiT import calcmultit as calcMt


if __name__ == '__main__':
    delta = np.zeros((4 , 2))
    delta[0 , 0] = 0.4
    delta[0 , 1] = 0.6
    delta[1 , 0] = 0.7
    delta[1 , 1] = 0.8
    delta[2 , 0] = 0.4
    delta[2 , 1] = 0.6
    delta[3 , 0] = 0.7
    delta[3 , 1] = 0.8

    data = np.loadtxt("D:\Roee\Downloads\temp\pydata.txt")
    labels = np.loadtxt("D:\Roee\Downloads\temp\pylabels.txt")
    mT = calcMt(data,labels)
    print(mT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpltlib
import pandas
import seaborn
from calcMultiT import calcmultit as calcMt


if __name__ == '__main__':
    # compare to data from matlab
    # data = np.loadtxt("D:\Roee\Downloads\temp\pydata.txt")
    # labels = np.loadtxt("D:\Roee\Downloads\temp\pylabels.txt")
    # mT = calcMt(data,labels)

    #create null distrubtion
    numobs = 50
    numfeaturs = 100
    numiter = 100
    randreal = np.empty(numiter)
    randefct = np.empty(numiter)

    # create null distribution
    for a in range(0,numiter):
        data = np.random.random([numobs,numfeaturs])
        labels = np.concatenate( (np.zeros([numobs/2,1]) , np.ones([numobs/2,1]) ) , axis=0)
        mT = calcMt(data,labels)
        temp = np.asarray(mT)
        randreal[a] = mT

    #create distribution with effect
    for a in range(0,numiter):
        data = np.random.random([numobs,numfeaturs])
        labels = np.concatenate( (np.zeros([numobs/2,1]) , np.ones([numobs/2,1]) ) , axis=0)
        idxlabelsA = np.where(labels == 0)
        data[idxlabelsA] = data[idxlabelsA]+1
        mT = calcMt(data,labels)
        temp = np.asarray(mT)
        randefct[a] = mT

    # plot the results
    font  = {'family' : 'normal',
             'weight' : 'bold',
             'size'   : 40}

    binBoundries = np.linspace(-5,25,50)
    plt.hist(randreal,bins=binBoundries,label = 'null distr')
    plt.hist(randefct,bins=binBoundries,label= 'real effect')
    plt.legend()
    plt.xlabel('count')
    plt.rc('font',**font)
    plt.ylabel('size of effect')
    plt.show()
    mpltlib.rcParams.update({'font.size': 40})

    plt.legend
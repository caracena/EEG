import numpy
import random
from scipy import stats
import matplotlib.pyplot as plt

def getTimeSeries(n,i,f):
    cont = []
    for x in range(n):
        cont.append(random.randint(i, f))
    return stats.zscore(cont)

def getBackward(seq,k):
    return numpy.diff(seq,n=k)

if __name__ == '__main__':

    a = getTimeSeries(100, 0, 10)

    x = []
    for k in range(50):  #order
        b  = getBackward(a,k)
        plt.plot(b)
        for t in range(len(b)):
            x.append((1 if b[t] >= 0 else 0,k,t))

    #ahora por d
    d = []
    for k in range(50):
        temp = []
        if x[1] == k:
            temp.append(x[0])
        sum = 0
        for i in range(len(temp)):
            sum = sum + (temp(i+1) - temp(i))^2
        d.append(sum)

    #debugging
    #plt.show()

    for p in d:
        print p

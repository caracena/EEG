from pylab import plot, hold, show
from scipy import sin, rand, arange
from pybrain.datasets            import SequenceClassificationDataSet
from pybrain.structure.modules   import LSTMLayer, SoftmaxLayer
from pybrain.supervised          import RPropMinusTrainer
from pybrain.tools.validation    import testOnSequenceData
from pybrain.tools.shortcuts     import buildNetwork


from .datasets import generateNoisySines

trndata = generateNoisySines(50, 40)
trndata._convertToOneOfMany( bounds=[0.,1.] )
tstdata = generateNoisySines(50, 20)
tstdata._convertToOneOfMany( bounds=[0.,1.] )

rnn = buildNetwork( trndata.indim, 5, trndata.outdim, hiddenclass=LSTMLayer, outclass=SoftmaxLayer, outputbias=False, recurrent=True)
#.. aa continuo manana

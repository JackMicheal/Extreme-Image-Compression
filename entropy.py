import tensorflow as tf
import numpy as np

from tensorflow.python.ops.init_ops import Constant
from tensorflow.python.ops.init_ops import Initializer
from tensorflow.python.ops.init_ops import Ones
from tensorflow.python.ops.init_ops import Orthogonal
from tensorflow.python.ops.init_ops import RandomNormal
from tensorflow.python.ops.init_ops import RandomUniform
from tensorflow.python.ops.init_ops import TruncatedNormal
from tensorflow.python.ops.init_ops import VarianceScaling
from tensorflow.python.ops.init_ops import Zeros
        
class entropy_estimator(tf.keras.layers.Layer):
    def __init__(self,filters_num = 3,K = 3, channels=24):
        super(entropy_estimator,self).__init__()
        #
        init_scale = 0.5
        
        filters = [filters_num for x in range(K)]
        self.filters = [1] + filters + [1]
        
        self.scale = init_scale ** (1.0/(len(self.filters)+1.0))
        
        self.likelihood_bound = 1e-9
        self.K = K
        self.channels = channels
        self.matrices = []
        self.factors = []
        self.bias = []
        #different channel follow same distribution
        #for i in range(K+1):
            #h >= 0, h = softplus(h)
        #    self.matrices.append(tf.random_normal(shape=[filters[i+1],filterd[i]],mean=0.0,stddev=1.0))
            #a >= -1, a = tanh(a)
        #    self.factors.append(tf.random_normal(shape=[filters[i+1],1],mean=0.0,stddev=1.0))
            #b
        #    self.bias.append(tf.random_normal(shape=[filters[i+1],1],mean=0.0,stddev=1.0))
        #different channel follow independent distribution
        
    def build(self,input_shape):
        filters = self.filters
        channels = self.channels
        for i in range(self.K+1):
            
            # init = log(...)
            init = np.log(np.expm1(1.0/self.scale/filters[i+1]))
            self.matrices.append(self.add_variable(name="matrix"+str(i),shape=[channels,filters[i+1],filters[i]],initializer=Constant(init)))
            
            # initializer = ... zero
            self.factors.append(self.add_variable(name="factor"+str(i),shape=[channels,filters[i+1],1],initializer=Zeros()))
            
            # RandomUniform (-0.5,0.5)
            self.bias.append(self.add_variable(name="bias"+str(i),shape=[channels,filters[i+1],1],initializer=RandomUniform(minval=-0.5,maxval=0.5)))                                       
            '''
            self.matrices.append(tf.Variable(tf.random_normal(shape=[channels,filters[i+1],filters[i]],mean=0.0,stddev=1.0)))
            self.factors.append(tf.Variable(tf.random_normal(shape=[channels,filters[i+1],1],mean=0.0,stddev=1.0)))
            self.bias.append(tf.Variable(tf.random_normal(shape=[channels,filters[i+1],1],mean=0.0,stddev=1.0)))
            '''
    def _logits_cumulative(self, inputs):
        logits = inputs
        for i in range(self.K+1):
            #print i,self.K
            #print self.matrices[i]
            logits = tf.matmul(tf.nn.softplus(self.matrices[i]),logits) + self.bias[i]
            logits = logits + tf.tanh(self.factors[i])*tf.tanh(logits)
        
        return logits
    
    def call(self,x):
        lower = self._logits_cumulative(x - 0.5)
        upper = self._logits_cumulative(x + 0.5)
        likelihoods = tf.sigmoid(upper) - tf.sigmoid(lower)
        
        #likelihood bound
        likelihoods = tf.maximum(likelihoods,self.likelihood_bound)
        
        return likelihoods

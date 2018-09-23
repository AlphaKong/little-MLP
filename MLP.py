# -*- coding: utf-8 -*-
'''
some parts references: http://neuralnetworksanddeeplearning.com/index.html
'''
import numpy as np
import readmnist

class next_batch_dataset(object):
    def __init__(self,data,label):
        self.data=data
        self.label=label
        self._num_examples=data.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._rtd=0
    
    
    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        '''
        如果遍历完一遍数据集，就将数据集打乱，然后进行第二次遍历
        '''
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.data = self.data[perm]
            self.label = self.label[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.data[start:end],self.label[start:end]

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)

class MLP_network(object):
    def __init__(self,sizes):
        #神经网络的层的参数
        self.num_layers=len(sizes)
        self.sizes=sizes
        
        #随机确定w和b，服从正态分布
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        self.weights=[np.random.randn(y,x)/np.sqrt(x) for x,y in zip(sizes[:-1],sizes[1:])]
        
        self.cost=CrossEntropyCost
        
        #数据集
        trX,trY,teX,teY=readmnist.read_data()
#        self.training_x=trX/255.0
#        self.training_y=trY
        self.training_x=trX[:50000]/255.0
        self.training_y=trY[:50000]
        self.validation_x=trX[50000:]/255.0
        self.validation_y=trY[50000:]
        self.test_x=teX/255.0
        self.test_y=teY
    
    #stochastic gradient descent
    #迭代周期，最小步长长度，学习率，测试数据
    def mini_batch_SGD(self,ephochs,mini_batch_size,lr):
        """Train the neural network using mini-batch stochastic
        gradient descent. The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs. The other non-optional parameters are
        self-explanatory. If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out. This is useful for
        tracking progress, but slows things down substantially."""
        
        n_test=self.test_x.shape[0]
        n=self.training_x.shape[0]
        
        for j in range(ephochs):
            next_batchs=next_batch_dataset(self.training_x,self.training_y)
            for i in range(int(n/mini_batch_size)):
                mini_batch=next_batchs.next_batch(mini_batch_size)
                self.update_mini_batch(mini_batch,lr)
#                print(i)
                
            r=self.evaluate(self.test_x,self.test_y)
            accuracy=r/n_test
            #print(r,n_test)
            print('Epoch{0}:{1}/{2} accuracy={3}'.format(j,r,n_test,accuracy))
#            np.random.shuffle(training_data)
#            #将数据集切分成等分的块
#            mini_batchs=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
#            #每一小块训练一次，就更新一次w,b
#            for mini_batch in mini_batchs:
#                self.update_mini_batch(mini_batch,lr)
#            if test_data:
#                r=self.evaluate(test_data)
#                accuracy=r/n_test
#                #print(r,n_test)
#                print('Epoch{0}:{1}/{2} accuracy={3}'.format(j,r,n_test,accuracy))
# 
#            else:
#                print('Epoch {0} complete'.format(j))
            
    
    #向前传播
    def feedforward(self,a):
        a=a.reshape(-1,1)
        for w,b in zip(self.weights,self.biases):
            a=sigmoid(np.dot(w,a)+b)
        return a
    
    def update_mini_batch(self,mini_batch,lr):
        """Update the network's weights and biases by applying
            gradient descent using backpropagation to a single mini batch.
            The "mini_batch" is a list of tuples "(x, y)", and "eta"
            is the learning rate."""
        #lr是学习率，learning rate
        #偏导数，偏差
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        for x,y in zip(mini_batch[0],mini_batch[1]):
            delta_nabla_b,delta_nabla_w=self.backprop(x,y) #BP算法，核心
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weights=[w-(lr/len(mini_batch))*nw
                      for w,nw in zip(self.weights,nabla_w)]
        self.biases=[b-(lr/len(mini_batch))*nb
                     for b,nb in zip(self.biases,nabla_b)]
    
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
          gradient for the cost function C_x.  ``nabla_b`` and
          ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
          to ``self.biases`` and ``self.weights``."""
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        #feedforward
        y=y.reshape(-1,1)
        x=x.reshape(-1,1)
        activation=x
        #list to store all the activations, layer by layer
        activations=[x]
        # list to store all the z vectors, layer by layer
        zs=[]
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigmoid(z)
#            print(activation)
            activations.append(activation)
        
        #backward
#        delta=self.cost_derivative(activations[-1],y)* \
#                sigmoid_prime(zs[-1])
        delta = (self.cost).delta(zs[-1], activations[-1], y)
#        print(delta.shape)
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for layer in range(2,self.num_layers):
            z=zs[-layer]
            sp=sigmoid_prime(z)
            delta=np.dot(self.weights[-layer+1].transpose(),delta)*sp
            nabla_b[-layer]=delta
            nabla_w[-layer]=np.dot(delta,activations[-layer-1].transpose())
        return (nabla_b,nabla_w)
        
    
    def evaluate(self, test_x,test_y):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        '''根据求得的参数，在testdata数据集测试，返回正确的结果'''
        #feedforward函数是根据所得的参数计算结果,x是测试集中的数据，
        #y是标签
        #numpy.argmax()返回最大值的下标
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                      for (x, y) in zip(test_x,test_y)]
        
        
        return sum(int(x == y) for (x, y) in test_results)
    
    #cost function derivative 代价函数的导数
    def cost_derivative(self, output_activations, y):
          """Return the vector of partial derivatives \partial C_x /
          \partial a for the output activations."""
          return (output_activations-y)
    #the cross-entropy
    def cross_entropy():
        pass
        
        
def sigmoid(z):
      """The sigmoid function."""
      return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    '''sigmoid的导数'''
    return sigmoid(z)*(1-sigmoid(z))  

if __name__=="__main__":
    mlp=MLP_network([784,100,10])
   
   
    mlp.mini_batch_SGD(30,10,0.3)
#    x=mlp.training_y[0]
#    print(x)
#    
#    trX,trY,teX,teY=readmnist.read_data()
#    next_batchs=next_batch_dataset(trX,trY)
#    xt=next_batchs.next_batch(8)
#    print(xt[0][0])
#    for x,y in zip(xt[0],xt[1]):
#        print(x.shape)
    


import os
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.utils import shuffle
from theano.tensor.nnet import conv2d
from scipy.io import loadmat
from theano.tensor.signal import pool

def init_weight_and_bias(M1, M2):
    M1 , M2 = int(M1) , int(M2)
    W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)


def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
    return w.astype(np.float32)


def error_rate(targets, predictions):
    return np.mean(targets != predictions)


def image2array(im):
    arr = np.array(im) # will be (H, W, 3)
    return arr.transpose((2, 1, 0))

def rearrange(X):
    # input is (32, 32, 3, N)
    # output is (N, 3, 32, 32)
    # N = X.shape[-1]
    # out = np.zeros((N, 3, 32, 32), dtype=np.float32)
    # for i in range(N):
    #     for j in range(3):
    #         out[i, j, :, :] = X[:, :, j, i]
    # return out / 255
    return (X.transpose(3, 2, 0, 1) / 255).astype(np.float32)

def get_data():
    if not os.path.exists('../large_files/train_32x32.mat'):
        print('Looking for ../large_files/train_32x32.mat')
        print('You have not downloaded the data and/or not placed the files in the correct location.')
        print('Please get the data from: http://ufldl.stanford.edu/housenumbers')
        print('Place train_32x32.mat and test_32x32.mat in the folder large_files adjacent to the class folder')
        exit()

    train = loadmat('../large_files/train_32x32.mat')
    test  = loadmat('../large_files/test_32x32.mat')
    return train, test

def getImageData():
    train , test = get_data()
    Xtrain = rearrange(train['X'])
    Ytrain = train['y'].flatten() - 1
    del train
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)

    Xtest  = rearrange(test['X'])
    Ytest  = test['y'].flatten() - 1
    del test
    return Xtrain , Ytrain , Xtest , Ytest


class HiddenLayer(object):
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W , b = init_weight_and_bias(M1, M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]

    def forward(self, X):
        return T.nnet.relu(X.dot(self.W) + self.b)


class ConvPoolLayer(object):
    def __init__(self, mi, mo, fw=5, fh=5, poolsz=(2, 2)):
        # mi = input feature map size
        # mo = output feature map size
        sz = (mo, mi, fw, fh)
        W0 = init_filter(sz, poolsz)
        self.W = theano.shared(W0)
        b0 = np.zeros(mo, dtype=np.float32)
        self.b = theano.shared(b0)
        self.poolsz = poolsz
        self.params = [self.W, self.b]

    def forward(self, X):
        conv_out = conv2d(input=X, filters=self.W)
        pooled_out = pool.pool_2d(
            input=conv_out,
            ws=self.poolsz,
            ignore_border=True
        )
        return T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))


class CNN(object):
    def __init__(self, convpool_layer_sizes, hidden_layer_sizes):
        self.convpool_layer_sizes = convpool_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y , Xvalid , Yvalid , lr=1e-4, mu=0.99, reg=1e-6, decay=0.99999, eps=1e-2, batch_sz=500, epochs=100, show_fig=True):
        lr = np.float32(lr)
        mu = np.float32(mu)
        reg = np.float32(reg)
        decay = np.float32(decay)
        eps = np.float32(eps)
        
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)
        Xvalid = Xvalid.astype(np.float32)
        Yvalid = Yvalid.astype(np.int32)
        # initialize convpool layers
        N, c, width, height = X.shape
        mi = c
        outw = width
        outh = height
        self.convpool_layers = []
        for mo, fw, fh in self.convpool_layer_sizes:
            layer = ConvPoolLayer(mi, mo, fw, fh)
            self.convpool_layers.append(layer)
            outw = (outw - fw + 1) / 2
            outh = (outh - fh + 1) / 2
            mi = mo

        # initialize mlp layers
        K = len(set(Y))
        self.hidden_layers = []
        M1 = self.convpool_layer_sizes[-1][0]*outw*outh # size must be same as output of last convpool layer
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1

        # logistic regression layer
        W, b = init_weight_and_bias(M1, K)
        self.W = theano.shared(W, 'W_logreg')
        self.b = theano.shared(b, 'b_logreg')

        # collect params for later use
        self.params = [self.W, self.b]
        for c in self.convpool_layers:
            self.params += c.params
        for h in self.hidden_layers:
            self.params += h.params

        # for momentum
        dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

        # for rmsprop
        cache = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

        # set up theano functions and variables
        thX = T.tensor4('X', dtype='float32')
        thY = T.ivector('Y')
        pY = self.forward(thX)

        rcost = reg*T.sum([(p*p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + rcost
        prediction = self.predict(thX)

        cost_predict_op = theano.function(inputs=[thX, thY], outputs=[cost, prediction])


        # momentum only
        updates = [
            (p, p + mu*dp - lr*T.grad(cost, p)) for p, dp in zip(self.params, dparams)
        ] + [
            (dp, mu*dp - lr*T.grad(cost, p)) for p, dp in zip(self.params, dparams)
        ]

        train_op = theano.function(
            inputs=[thX, thY],
            updates=updates
        )

        n_batches = N // batch_sz
        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                train_op(Xbatch, Ybatch)

                if j % 20 == 0:
                    c, p = cost_predict_op(Xvalid, Yvalid)
                    costs.append(c)
                    e = error_rate(Yvalid, p)
                    print("i:", i+1, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for c in self.convpool_layers:
            Z = c.forward(Z)
        Z = Z.flatten(ndim=2)
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return T.nnet.softmax(Z.dot(self.W) + self.b)

    def predict(self, X):
        pY = self.forward(X)
        return T.argmax(pY, axis=1)


def main():
    Xtrain, Ytrain , Xtest , Ytest = getImageData()
    model = CNN(
        convpool_layer_sizes=[(20, 5, 5), (20, 5, 5)],
        hidden_layer_sizes=[500 , 300],
    )
    model.fit(Xtrain, Ytrain , Xtest , Ytest)

if __name__ == '__main__':
    main()

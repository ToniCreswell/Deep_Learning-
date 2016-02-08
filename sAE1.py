#A Denoising Sparse Autoencoder class using THEANO
#Class also has the encoder, decoder and getUpdate function to use for training
#The decoding weights are NOT transposed versions of the encoding weights


from theano.tensor.shared_randomstreams import RandomStreams
from theano import tensor as T 
import numpy as np
import os

import theano
theano.config.floatX='float32'

class AE(object):
	
	def __init__(
		self, 
		numpy_rndgen,
		theano_rndgen,
		input=None,
		n_in=30*30,
		n_hidden=100,
		Wenc=None,
		Wdec=None,
		benc=None,
		bdec=None
		):

			self.n_in=n_in
			self.n_hidden=n_hidden

			#Theano random num generator --> symbolic random numbers:
			if not theano_rndgen:
				theano_rndgen=RandomStreams(numpy_rndgen.randint(2**30))

			self.theano_rndgen=theano_rndgen
			
			#Initialise the Wenc and Wdec with small random vars:
			if Wenc is None:
				initial_Wenc = np.asarray(
					numpy_rndgen.uniform(
						low=-0.01, high=0.01, size=(n_in,n_hidden)
					),
					dtype=theano.config.floatX
				)
				Wenc=theano.shared(value=initial_Wenc, name='Wenc', borrow=True)
			if Wdec is None:
				initial_Wdec = np.asarray(
					numpy_rndgen.uniform(
						low=-0.01, high=0.01, size=(n_hidden,n_in)
					),
					dtype=theano.config.floatX
				)
				Wdec=theano.shared(value=initial_Wdec, name='Wdec', borrow=True)
			#Init the benc and bdec with zeros:
			if benc is None:
				benc=theano.shared(
					value=np.zeros(
						n_hidden,
						dtype=theano.config.floatX
					),
					name='benc',
					borrow=True
				)
			if bdec is None:
				bdec=theano.shared(
					value=np.zeros(
						n_in,
						dtype=theano.config.floatX
					),
					name='bdec',
					borrow=True
				)

			#set all the values:
			self.Wenc=Wenc
			print 'Wenc:',self.Wenc.dtype
			self.Wdec=Wdec
			print 'Wdec:',self.Wdec.dtype
			self.benc=benc
			print 'benc:',self.benc.dtype
			self.bdec=bdec
			print 'bdec:',self.bdec.dtype

			print 'Wenc:', self.Wenc.shape.eval(), 'Wdec:', self.Wdec.shape.eval()
			print 'benc:', self.benc.shape.eval(), 'bdec:', self.bdec.shape.eval()

			if input is None:
				self.x=T.dmatrix(name='input')
			else:
				self.x=input
				print 'input:', input.dtype

			#Parmas
			self.params=[self.Wenc, self.benc, self.Wdec, self.bdec]

	def encode(self,input_x):
		print('encoding...')
		return T.nnet.sigmoid(T.dot(input_x,self.Wenc) + self.benc)

	def decode(self,encoded):
		print('decoding...')
		return T.nnet.sigmoid(T.dot(encoded,self.Wdec) + self.bdec)

	def addNoise(self,input, noiseLevel):
		return self.theano_rndgen.binomial(size=input.shape, n=1, p=1-noiseLevel,dtype=theano.config.floatX)*input

	def getUpdate(self, noiseLevel, learnRate, roh, beta):

		#Add some corruption...
		if(noiseLevel>0):
			noiseySample=self.addNoise(self.x, noiseLevel)

		y=self.encode(noiseySample)
		# mean activation
		roh_hat=T.mean(y, axis=0) #Mean activation across samples in a  batch (roh_hat should have dim 1,100 )

		z=self.decode(y)


		
		#Sparisity parameter
		sparsity=((roh * T.log(roh / roh_hat)) + ((1 - roh) * T.log((1 - roh) / (1 - roh_hat)))).sum()

		L = - T.sum(((self.x * T.log(z)) + ((1 - self.x) * T.log(1 - z))), axis=1) #Log likely hood over the samples (N.B. L is a vector)
		
		E=((self.x-z)**2).sum()
		
		cost= T.mean(L) + beta * sparsity

		grads= T.grad(cost,self.params)

		updates=[(param, param - learnRate*grad) for param, grad in zip(self.params, grads)]

		print updates[1][1].dtype

		return (cost, updates, roh_hat, T.mean(E))




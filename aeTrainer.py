from sAE1 import AE 
import theano
import numpy as np
from theano import tensor as T

def trainer(train_set,myAE,noiseLevel,learnRate,roh,beta,max_epoch,batchSize):

	#train_set  :  shared var of training samples
	#myAE		:  AE of type sAE1
	#noiseLevel :  float % of noise to add to the image
	#learnRate  :  float learning rate used in SGD
	#roh 		:  float > 1 ideal sparsity
	#beta   	:  weighting of the sparsity term in the cost function
	#max_epoch	:  No of epoches
	#batchSize 	:  Size of batches used to train the net N.B. should be factor of the no of training samples
	

	#Symbolic vars
	index = T.lscalar()    # index to a [mini]batch
	x = T.matrix('x') 
	
	print(type(myAE))
	print(type(train_set))
	print(train_set.shape.eval())
	
	cost, updates, roh_hat = myAE.getUpdate(
	noiseLevel=noiseLevel,
	learnRate=learnRate,
	roh=roh,
	beta=beta,
	)

	trainAE = theano.function(
		inputs=[index],
		outputs=[cost, roh_hat],
		updates=updates,
		givens={
			x: train_set[index*batchSize:(index+1)*batchSize]  #N.B. has shape (batch_size x 784)
			}
		)

	#Compute no of mini-batches:
	n_batches=train_set.get_value(borrow=True).shape[0]/batchSize

	minCost='Inf'
	print 'Epoch \t Cost \t Roh_hat'
	for epoch in xrange(max_epoch):
		c=[]
		r=[]
		for batch_idx in xrange(n_batches):
			cost, roh_hat=trainAE(batch_idx)
			c.append(cost)
			r.append(roh_hat)
		if(np.mean(c)<minCost):
			minCost=np.mean(c)
			print epoch+1, '\t',  bcolors.OKGREEN , np.mean(c) , bcolors.ENDC, '\t', np.mean(roh_hat)
		else:
			print epoch+1, '\t', np.mean(c), '\t', np.mean(roh_hat)
	return myAE
	
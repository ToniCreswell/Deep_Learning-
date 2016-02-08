See Wiki For more... https://github.com/ToniCreswell/Deep_Learning-/wiki

Quick Example: https://github.com/ToniCreswell/Deep_Learning-/blob/master/sAE_example.ipynb

This repo (will) contain(s) build blocks for deep learning.

Current Contributions:
sAE1.py - Spase Denoising Auto-Encoder (sAE)
aeTrainer - Function to train the sAE
sAE_example.ipynb - ipython notebook to show how the code can be used
requirements.txt - libs used, tho I suggest using the instructions below to install them


Using the sAE code:

Download data from:
http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

(Optional) Create a Venv & activate:
$ virtualenv venv
$ source venv/bin/activate

(Download dependancies):
$ pip install numpy
$ pip install matplotlib==1.4.1
$ pip install os

$ git clone git://github.com/Theano/Theano.git
$ cd Theano
$ python setup.py develop

To use the ipython notebook
$ pip install ipython
$ pip install jupyter

cd to your working dir...

$ ipython notebook

This will start a web browser

Go to the sAE_example.ipynb to get started...

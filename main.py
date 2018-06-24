import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from perceptron import Perceptron
import pdr

def __main__:

	# get the iris data
	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
		header = None)

	# Plot 100 samples od the data

	y = dif.iloc[0:100, 4].values
	y = np.where(y == 'Iris-setosa', -1, 1)
	X = dif.iloc[0:100, [0, 2]].values

	plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa')
	plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 'x', label = 'versicolor')
	plt.xlabel('petal length')
	plt.ylabel('sepal length')
	plt.legend(loc = 'upper left')
	plt.show()

	# get the perceptron model
	model = Perceptron(eta = 0.1, n_iter = 10)

	# train the model
	model.fit(X, y)

	# plot the training error
	plt.plot(range(1, len(model.errors_) + 1), model.errors_, marker = 'o')
	plt.xlabel('Epochs')
	plt.ylabel('Number of misclassifications')
	plt.show()

	# create decision regions
	pdr.plot_decision_regions(X, y, classifier = model)
	plt.xlabel('sepal length [cm]')
	plt.ylabel('petal length [cm]')
	plt.legend(loc = 'upper left')
	plt.show()


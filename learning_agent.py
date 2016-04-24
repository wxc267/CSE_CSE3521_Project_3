import math
import numpy


class LogisticRegressionLearningAgent:
	
	def __init__(self, dimension=None, model=None):
		if model is None and dimension is None:
			raise Exception()
		if model is None:
			model = [0] * dimension
		if dimension is None:
			dimension = len(model)
		if dimension != len(model):
			raise Exception()
		self.dimension = dimension
		self.weight_vector = numpy.array(model)
		self.initial_learning_rate = 10**-6
		self.time = 0
		
	@classmethod
	def gradient_L(cls, w, x, y):
		logistic = 1 + math.exp(-y * numpy.dot(w, x))
		gradient = -y * x * ((logistic - 1) /logistic)
		return gradient
	
	@property
	def next_learning_rate(self):
		self.time += 1
		return self.initial_learning_rate / self.time
		
	def learn(self, feature, label):
		w = self.weight_vector
		x = numpy.array(feature)
		y = -1 if label == 0 else 1
		gradient = self.gradient_L(w, x, y)
		self.weight_vector = w - self.next_learning_rate * gradient
		
	def test(self, feature):
		result = numpy.dot(self.weight_vector, numpy.array(feature))
		return 1 if result > 0 else 0
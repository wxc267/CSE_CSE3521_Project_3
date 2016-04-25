import sys
from logistic_regression import *

def read_command_args():
	args = sys.argv
	directory = './data/'
	if len(args) != 6:
		raise ValueError('5 arguments expected!')
	else:
		try:
			training_feature = directory + args[1]
			training_label = directory + args[2]
			model_file = directory + args[3]
			feature_dimension = int(args[4])
			feature_length = int(args[5])
		except:
			raise Exception('Bad arguments!')
		return training_feature, training_label, model_file, feature_dimension, feature_length

training_feature, training_label, model_file, feature_dimension, feature_length = read_command_args()
agent = train_from_file(training_feature, training_label, feature_dimension, feature_length)
model = agent.weight_vector
export_vector(model, model_file)
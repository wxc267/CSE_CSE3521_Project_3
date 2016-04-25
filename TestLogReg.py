import sys
from logistic_regression import *

def read_command_args():
	args = sys.argv
	directory = './data/'
	if len(args) != 5:
		raise ValueError('4 arguments expected!')
	else:
		try:
			model_file = directory + args[1]
			test_feature = directory + args[2]
			predicted_label = directory + args[3]
			feature_dimension = int(args[4])
		except:
			raise Exception('Bad arguments!')
		return model_file, test_feature, predicted_label, feature_dimension

model_file, test_feature, predicted_label, feature_dimension = read_command_args()
test_results = test_from_file(test_feature, model_file)
export_vector(test_results, predicted_label, orientation='column')
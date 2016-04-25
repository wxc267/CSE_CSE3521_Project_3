import sys
from logistic_regression import *

def read_command_args():
	args = sys.argv
	directory = './data/'
	if len(args) != 3:
		raise ValueError('2 arguments expected!')
	else:
		try:
			predicted_label = directory + args[1]
			true_label = directory + args[2]
		except:
			raise Exception('Bad arguments!')
		return predicted_label, true_label

predicted_label, true_label = read_command_args()
score, count = check_accuracy(predicted_label, true_label)
print('score:', score)
print('accuracy: {}%'.format(100 * score / count))
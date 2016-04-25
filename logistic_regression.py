from learning_agent import *

def str_to_list(string, list_type):
	return [list_type(x) for x in string.split()]

def train_from_file(feature_file, label_file, feature_dimension, feature_length):
	with open(feature_file) as features, open(label_file) as labels:
		agent = LogisticRegressionLearningAgent(dimension=feature_dimension)
		for raw_feature, raw_label, i in zip(features, labels, range(feature_length)):
			feature = str_to_list(raw_feature, int)
			label = int(raw_label)
			agent.learn(feature, label)
		return agent

def test_from_file(feature_file, model_file):
	with open(feature_file) as features, open(model_file) as model:
		weight_vector = str_to_list(model.readline(), float)
		agent = LogisticRegressionLearningAgent(model=weight_vector)
		predicted_labels = []
		for feature in features:
			predicted_labels += [agent.test(str_to_list(feature, int))]
		return predicted_labels

def check_accuracy(predicted_label_file, true_label_file):
	score, count = 0, 0
	with open(predicted_label_file) as pred_labels, open(true_label_file) as true_labels:
		for pred_label, true_label in zip(pred_labels, true_labels):
			count += 1
			if int(pred_label) == int(true_label):
				score += 1
	return score, count

def export_vector(vector, path, orientation='row'):
	end_char = ' ' if orientation == 'row' else '\n' if orientation == 'column' else ''
	with open(path, mode='w') as fout:
		for value in vector:
			print(value, file=fout, end=end_char) 

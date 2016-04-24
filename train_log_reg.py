from learning_agent import *

def str_to_list(string, list_type):
	return [list_type(x) for x in string.split()]

def train_from_file(feature_file, label_file, feature_dimension):
	with open(feature_file) as features, open(label_file) as labels:
		agent = logistic_regression_learning_agent(dimension=feature_dimension)
		for raw_feature, raw_label in zip(features, labels):
			feature = str_to_list(raw_feature, int)
			label = int(raw_label)
			agent.learn(feature, label)
		return agent

def test_from_file(feature_file, model_file):
	with open(feature_file) as features, open(model_file) as model:
		weight_vector = str_to_list(model.readline(), float)
		agent = logistic_regression_learning_agent(model=weight_vector)
		predicted_labels = []
		for feature in features:
			predicted_labels += [agent.test(str_to_list(feature, int))]
		return predicted_labels

def export_vector(vector, path, orientation='row'):
	end_char = ' ' if orientation == 'row' else '\n' if orientation == 'column' else ''
	with open(path, mode='w') as fout:
		for value in vector:
			print(value, file=fout, end=end_char) 
			
training_feature, training_label, model_file = './trainingFeature.dat', './trainingLabel.dat', './modelFile.dat'
feature_dimension = 785
agent = train_from_file(training_feature, training_label, feature_dimension)
model = agent.weight_vector
export_vector(model, model_file)

test_feature, test_label = './testFeature.dat', './testLabel.dat'
predicted_labels = test_from_file(test_feature, model_file)
export_vector(predicted_labels, test_label, orientation='column')
	
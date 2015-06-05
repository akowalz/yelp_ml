import sklearn
import pickle
import pdb
from sklearn import svm
from sklearn.externals import joblib


def validate(inputs, stars, ratings, classifier, num_folds=10):
	data_size = len(inputs)
	size_fold = data_size / num_folds
	for fold in range(num_folds):
		training_inputs = inputs
		training_stars = stars
		training_ratings = ratings
		validation_inputs = inputs[(fold*size_fold):((fold+1)*size_fold)] # produce validation set
		validation_stars = stars[(fold*size_fold):((fold+1)*size_fold)]
		valudation_ratings = ratings[(fold*size_fold):((fold+1)*size_fold)]
		training_inputs[(fold*size_fold):((fold+1)*size_fold)] = [] # remove the validation set from the training data
		training_stars[(fold*size_fold):((fold+1)*size_fold)] = [] # remove the validation set from the training data
		training_ratings[(fold*size_fold):((fold+1)*size_fold)] = [] # remove the validation set from the training data

		lsvc.fit(training_inputs, training_stars)

		correct_predictions = 0
		for i in range(len(validation_inputs)):
			pdb.set_trace()
			correct_star = '{:.1f}'.format(validation_stars[i])
			predicted_star = '{:.1f}'.format(classifier.predict(validation_inputs[i])[0])
			print 'Predicted {}, got {}'.format(predicted_star, correct_star)
			if correct_star == predicted_star:
				correct_predictions += 1

		print 'Correctly predicted {} out of {}, {}%'.format(correct_predictions, len(validation_inputs), (float(correct_predictions)/float(len(validation_inputs)))*100)

lsvc = svm.LinearSVC()
svc = svm.SVC()

with open('pickles/inputs', 'r') as f:
	inputs = pickle.load(f)
with open('pickles/ratings_inputs', 'r') as f:
	ratings = pickle.load(f)
with open('pickles/star_inputs', 'r') as f:
	stars = pickle.load(f)

validate(inputs, stars, ratings, lsvc, 10)

import models_partc
import models_partb
from sklearn.model_selection import KFold, ShuffleSplit
import numpy as np
from numpy import mean
from sklearn.linear_model import LogisticRegression
import utils
from sklearn.metrics import *

RANDOM_STATE = 1234567

def get_acc_auc_kfold(X,Y,k=5):
	#First, to get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
	kfold = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

	accuracy_list = []
	auc_list = []

	for train, test in kfold.split(X):
		# split data into train and test sets
		X_train, X_test = X[train], X[test]
		Y_train, Y_test = Y[train], Y[test]

		# make the prediction on the test data
		Y_predict = models_partc.logistic_regression_pred(X_train, Y_train, X_test)
		accuracy = accuracy_score(Y_predict, Y_test)
		auc = roc_auc_score(Y_predict, Y_test)
		accuracy_list.append(accuracy)
		auc_list.append(auc)

	accuracy_mean = np.mean(accuracy_list)
	auc_mean = np.mean(auc_list)

	return accuracy_mean, auc_mean

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
	shuffle_split = ShuffleSplit(n_splits=iterNo, test_size=test_percent, random_state=RANDOM_STATE)
	accuracy_list = []
	auc_list = []

	for train, test in shuffle_split.split(X):
		# split data into train and test sets
		X_train, X_test = X[train], X[test]
		Y_train, Y_test = Y[train], Y[test]

		# make the prediction on the test data
		Y_predict = models_partc.logistic_regression_pred(X_train, Y_train, X_test)
		accuracy = accuracy_score(Y_predict, Y_test)
		auc = roc_auc_score(Y_predict, Y_test)
		accuracy_list.append(accuracy)
		auc_list.append(auc)

	accuracy_mean = np.mean(accuracy_list)
	auc_mean = np.mean(auc_list)

	return accuracy_mean, auc_mean

def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print("Classifier: Logistic Regression__________")
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print(("Average Accuracy in KFold CV: "+str(acc_k)))
	print(("Average AUC in KFold CV: "+str(auc_k)))
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print(("Average Accuracy in Randomised CV: "+str(acc_r)))
	print(("Average AUC in Randomised CV: "+str(auc_r)))

if __name__ == "__main__":
	main()


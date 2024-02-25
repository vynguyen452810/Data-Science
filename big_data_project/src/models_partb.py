import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn import tree

import utils

RANDOM_STATE = 1234567

def logistic_regression_pred(X_train, Y_train):
	logreg = LogisticRegression()
	logreg.fit(X_train, Y_train)
	y_predict = logreg.predict(X_train)
	return y_predict

def svm_pred(X_train, Y_train):
	linear = LinearSVC()
	linear.fit(X_train, Y_train)
	y_predict = linear.predict(X_train)
	return y_predict

def decisionTree_pred(X_train, Y_train):
	np.random.seed(RANDOM_STATE)
	decision_tree = DecisionTreeClassifier(max_depth=5)
	decision_tree.fit(X_train, Y_train)
	y_predict = decision_tree.predict(X_train)
	return y_predict


def classification_metrics(Y_pred, Y_true):
	accuracy = accuracy_score(Y_pred, Y_true)
	auc = roc_auc_score(Y_pred, Y_true)
	precision = precision_score(Y_pred, Y_true)
	recall = recall_score(Y_pred, Y_true)
	f1score = f1_score(Y_pred, Y_true)

	return accuracy, auc, precision, recall, f1score

def display_metrics(classifierName,Y_pred,Y_true):
	print("______________________________________________")
	print(("Classifier: "+classifierName))
	acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
	print(("Accuracy: "+str(acc)))
	print(("AUC: "+str(auc_)))
	print(("Precision: "+str(precision)))
	print(("Recall: "+str(recall)))
	print(("F1-score: "+str(f1score)))
	print("")

def main():
	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	
	display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train),Y_train)
	display_metrics("SVM",svm_pred(X_train,Y_train),Y_train)
	display_metrics("Decision Tree",decisionTree_pred(X_train,Y_train),Y_train)
	
if __name__ == "__main__":
	main()
	

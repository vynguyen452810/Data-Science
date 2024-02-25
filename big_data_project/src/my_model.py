import utils
import models_partc
import etl
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import *
#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

RANDOM_STATE = 1234567
def my_features():

	events_test = pd.read_csv('../data/test/events.csv')
	feature_map_test = pd.read_csv('../data/test/event_feature_map.csv')
	dump_path = '../dump/'
	
	# aggregate events
	selected_colums = ['patient_id', 'event_id', 'value']
	events_test = events_test[selected_colums]
	aggregated_events_test = etl.aggregate_events(events_test, None, feature_map_test, dump_path)
	
	# create patient feature
	temp_patient_features = aggregated_events_test
	temp_patient_features['zip'] = list(zip(aggregated_events_test['feature_id'], aggregated_events_test['feature_value']))
	temp_patient_features = temp_patient_features.drop(['feature_id','feature_value'], axis = 1)

	patient_features = aggregated_events_test.groupby('patient_id').apply(lambda x: list(zip(x['feature_id'], x['feature_value']))).to_dict()

	# save_svmlight
	deliverable1 = open('../deliverables/test_features.txt', 'wb')
	deliverable2 = open('../deliverables/test_features.train', 'wb')

	# sort by patient_id
	for patient_id, features in sorted(patient_features.items()):
		txt = '{} '.format(patient_id)
		train = '0 '

		for feature in sorted(features):
			txt += '{}:{:.6f} '.format(int(feature[0]), feature[1])
			train += '{}:{:.6f} '.format(int(feature[0]), feature[1])
		txt += '\n'
		train += '\n'

		deliverable1.write(bytes((txt),'UTF-8')); #Use 'UTF-8'
		deliverable2.write(bytes((train),'UTF-8'))

	# get the X_test data
	X_test, Y_test = utils.get_data_from_svmlight("../deliverables/test_features.train")
	# get the train data from features_svmlight.train
	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	
	return X_train, Y_train, X_test, Y_test


def random_forest_prediction(X_train,Y_train,X_test):
	classifier = RandomForestClassifier()
	classifier.fit(X_train, Y_train)
	y_predict = classifier.predict(X_test)
	return y_predict

def gradient_boosting_prediction(X_train,Y_train,X_test):
	classifier = GradientBoostingClassifier()
	classifier.fit(X_train, Y_train)
	y_predict = classifier.predict(X_test)
	return y_predict

def randomforest_kfold(X,Y,k=5):
	kfold = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

	accuracy_list = []
	auc_list = []

	for train, test in kfold.split(X):
		X_train, X_test = X[train], X[test]
		Y_train, Y_test = Y[train], Y[test]

		# make the prediction on the test data
		Y_predict = random_forest_prediction(X_train, Y_train, X_test)
		accuracy = accuracy_score(Y_predict, Y_test)
		auc = roc_auc_score(Y_predict, Y_test)
		accuracy_list.append(accuracy)
		auc_list.append(auc)

	accuracy_mean = np.mean(accuracy_list)
	auc_mean = np.mean(auc_list)

	return accuracy_mean, auc_mean

def gradientboosting_kfold(X,Y,k=5):
	kfold = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

	accuracy_list = []
	auc_list = []

	for train, test in kfold.split(X):
		# split data into train and test sets
		X_train, X_test = X[train], X[test]
		Y_train, Y_test = Y[train], Y[test]

		# make the prediction on the test data
		Y_predict = gradient_boosting_prediction(X_train, Y_train, X_test)
		accuracy = accuracy_score(Y_predict, Y_test)
		auc = roc_auc_score(Y_predict, Y_test)
		accuracy_list.append(accuracy)
		auc_list.append(auc)

	accuracy_mean = np.mean(accuracy_list)
	auc_mean = np.mean(auc_list)

	return accuracy_mean, auc_mean

def randomforest_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	shuffle_split = ShuffleSplit(n_splits=iterNo, test_size=test_percent, random_state=RANDOM_STATE)
	accuracy_list = []
	auc_list = []

	for train, test in shuffle_split.split(X):
		# split data into train and test sets
		X_train, X_test = X[train], X[test]
		Y_train, Y_test = Y[train], Y[test]

		# make the prediction on the test data
		Y_predict = random_forest_prediction(X_train, Y_train, X_test)
		accuracy = accuracy_score(Y_predict, Y_test)
		auc = roc_auc_score(Y_predict, Y_test)
		accuracy_list.append(accuracy)
		auc_list.append(auc)

	accuracy_mean = np.mean(accuracy_list)
	auc_mean = np.mean(auc_list)

	return accuracy_mean, auc_mean

def gradientboosting_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	shuffle_split = ShuffleSplit(n_splits=iterNo, test_size=test_percent, random_state=RANDOM_STATE)
	accuracy_list = []
	auc_list = []

	for train, test in shuffle_split.split(X):
		# split data into train and test sets
		X_train, X_test = X[train], X[test]
		Y_train, Y_test = Y[train], Y[test]

		# make the prediction on the test data
		Y_predict = gradient_boosting_prediction(X_train, Y_train, X_test)
		accuracy = accuracy_score(Y_predict, Y_test)
		auc = roc_auc_score(Y_predict, Y_test)
		accuracy_list.append(accuracy)
		auc_list.append(auc)

	accuracy_mean = np.mean(accuracy_list)
	auc_mean = np.mean(auc_list)

	return accuracy_mean, auc_mean

def main():
	X_train, Y_train, X_test, Y_test = my_features()
	# X, Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	Y_pred = random_forest_prediction(X_train,Y_train,X_test)
	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

	print("Classifier: Random Forest__________")
	acc_k,auc_k = randomforest_kfold(X_train,Y_train)
	print(("Average Accuracy in KFold CV: "+str(acc_k)))
	print(("Average AUC in KFold CV: "+str(auc_k)))
	acc_r,auc_r = randomforest_randomisedCV(X_train,Y_train)
	print(("Average Accuracy in Randomised CV: "+str(acc_r)))
	print(("Average AUC in Randomised CV: "+str(auc_r)))

	print("Classifier: Gradient Boosting__________")
	acc_k,auc_k = gradientboosting_kfold(X_train,Y_train)
	print(("Average Accuracy in KFold CV: "+str(acc_k)))
	print(("Average AUC in KFold CV: "+str(auc_k)))
	acc_r,auc_r = gradientboosting_randomisedCV(X_train,Y_train)
	print(("Average Accuracy in Randomised CV: "+str(acc_r)))
	print(("Average AUC in Randomised CV: "+str(auc_r)))
if __name__ == "__main__":
    main()

	
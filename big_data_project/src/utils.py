import pandas as pd
from datetime import datetime
from datetime import timedelta
from sklearn.datasets import load_svmlight_file


def date_offset(x,no_days):
    return datetime.strptime(x, '%Y-%m-%d') + timedelta(days=no_days)
   
def date_convert(x):
    return datetime.strptime(x, '%Y-%m-%d')
  
def bag_to_svmlight(input):
    return ' '.join(( "%d:%f" % (fid, float(fvalue)) for fid, fvalue in input))


def get_data_from_svmlight(svmlight_file):
    data_train = load_svmlight_file(svmlight_file,n_features=3190)
    X_train = data_train[0]
    Y_train = data_train[1]
    return X_train, Y_train
 
def generate_submission(svmlight_with_ids_file, Y_pred):
    try:
        f = open(svmlight_with_ids_file)
        lines = f.readlines()

        # Check if the length of Y_pred matches the number of lines in the file
        if len(lines) != len(Y_pred):
            raise ValueError("Number of predictions does not match the number of data points.")

        target = open('../deliverables/my_predictions.csv', 'w')
        target.write("%s,%s\n" % ("patient_id", "label"))

        for i in range(len(lines)):
            target.write("%s,%s\n" % (str(lines[i].split()[0]), str(Y_pred[i])))
        
        target.close()
        f.close()

    except FileNotFoundError:
        print("Input file not found.")
    except Exception as e:
        print("An error occurred:", str(e))

def get_filter_dead_alive(events, mortality):
    patient_id = events['patient_id'].unique()
    dead_id = mortality['patient_id']
    alive_id = pd.Series(list(set(patient_id) - set(dead_id)))

    dead_events = events[events.patient_id.isin(dead_id)]
    alive_events = events[events.patient_id.isin(alive_id)]

    return dead_events, alive_events



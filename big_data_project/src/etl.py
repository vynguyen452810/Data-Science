import numpy as np
import pandas as pd
from datetime import timedelta
import utils

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    '''
    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    event_dates = events.loc[:, ['patient_id', 'timestamp']]
    
    # split events into two groups based on whether the patient is alive or deceased
    death_dates = mortality.loc[:, ['patient_id', 'timestamp']]
    # convert data in timestamp collumn from string to datetime objects and substarct 30 days window
    death_dates['timestamp'] = pd.to_datetime(death_dates['timestamp']) - timedelta(days=30)


    alive_dates = event_dates.loc[~event_dates['patient_id'].isin(mortality['patient_id'])]
    alive_dates = alive_dates.groupby(['patient_id']).max().reset_index()

    #indx_date = alive_dates.append(death_dates).reset_index(drop=True)
    indx_date = pd.concat([alive_dates, death_dates], ignore_index=True)
    indx_date = indx_date.rename(columns = {'timestamp':'indx_date'})

    csv_filename = 'etl_index_dates.csv'
    indx_date.to_csv(deliverables_path + csv_filename, columns=['patient_id', 'indx_date'], index=False)

    return indx_date

def filter_events(events, indx_date, deliverables_path):
    
    '''

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    # convert timestamp to data object
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    indx_date['indx_date'] = pd.to_datetime(indx_date['indx_date'])
    # merge 2 df: events and indx_date using share collumn 'patient_id'
    merged_df = pd.merge(events,indx_date, on = 'patient_id')

    upper_bound = merged_df['indx_date']
    lower_abound = merged_df['indx_date'] - timedelta(days = 2000)

    filter_events = merged_df[(merged_df['timestamp'] <= upper_bound) & (merged_df['timestamp'] >= lower_abound)]
    filter_events = filter_events[['patient_id','event_id','value']]

    csv_filename = 'etl_filtered_events.csv'
    filter_events.to_csv(deliverables_path + csv_filename,columns=['patient_id', 'event_id', 'value'], index=False)
    return filter_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    # 1. Replace event_id's with index available in event_feature_map.csv
    aggregated_events_df = pd.merge(filtered_events_df, feature_map_df, on='event_id')

    # 2. Remove events with n/a values
    aggregated_events_df = aggregated_events_df.dropna()

    # 3. Aggregate events using sum and count to calculate feature value
    aggregated_events_df['Type'] = aggregated_events_df['event_id'].str.contains('DRUG|DIAG')

    d_events = aggregated_events_df[aggregated_events_df['Type'] == True].groupby(['patient_id', 'idx'])['value'].sum()
    l_events = aggregated_events_df[aggregated_events_df['Type'] == False].groupby(['patient_id', 'idx'])['value'].count()
    d_events = d_events.to_frame(name='feature_value').reset_index()
    l_events = l_events.to_frame(name='feature_value').reset_index()

    # Combine the aggregated event DataFrames
    aggregated_events_df = pd.concat([d_events, l_events])
    aggregated_events_df.rename(columns={'idx': 'feature_id'}, inplace=True)

    # 6.Normalization (within each feature)
    max_values = aggregated_events_df.groupby(['feature_id'])['feature_value'].transform('max')
    aggregated_events_df['feature_value'] = aggregated_events_df['feature_value'] / max_values

    # Save to CSV
    aggregated_events_df.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    return aggregated_events_df


def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''

    # for patient_features{key: patient_id -- value: array of tuples(feature_id, feature_value)}
    patient_group = aggregated_events.groupby(['patient_id'])

    def sort_and_create_tuple(group):
        # sort by 'feature_id'
        sorted_group = group.sort_values('feature_id')
        # create a list of tuples for 'feature_id' and 'feature_value'
        tuples = [(row['feature_id'], row['feature_value']) for _, row in sorted_group.iterrows()]
        return tuples
    
    patient_tuple_dict = patient_group.apply(sort_and_create_tuple)
    patient_features_dict = patient_tuple_dict.to_dict()

    # for mortality{key:patient_id, value: label}
    mortality = mortality.drop('timestamp', axis=1)
    mortality_dict = dict(zip(mortality['patient_id'], mortality['label']))
    return patient_features_dict, mortality_dict


def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    
    
    for i in patient_features:
        features = pd.DataFrame(patient_features[i]).sort_values(0).values.tolist()
        svmlight_format = str(mortality.get(i, 0)) + " " + utils.bag_to_svmlight(features) + " \n"
        features_format = str(int(i))+" "+str(mortality.get(i, 0))+ " " + utils.bag_to_svmlight(features) + " \n"

        deliverable1.write(bytes((svmlight_format),'UTF-8')) #Use 'UTF-8'
        deliverable2.write(bytes((features_format),'UTF-8'))

    deliverable1.close()
    deliverable2.close()
        

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

    calculate_index_date(events, mortality, feature_map)
if __name__ == "__main__":
    main()
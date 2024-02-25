import time
import pandas as pd
import numpy as np
import utils


def read_csv(filepath):
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')
    return events, mortality


def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    dead_events, alive_events = utils.get_filter_dead_alive(events, mortality)

    # aggregate data - group rows of df based on the values in one or more columns
    grouped_alive = alive_events.groupby('patient_id')
    grouped_dead = dead_events.groupby('patient_id')

    # Calculate unqiue event count for each group
    alive_count = grouped_alive['event_id'].count()
    dead_count = grouped_dead['event_id'].count()

    # compute stats
    avg_dead_event_count = dead_count.mean()
    max_dead_event_count = dead_count.max()
    min_dead_event_count = dead_count.min()
    avg_alive_event_count = alive_count.mean()
    max_alive_event_count = alive_count.max()
    min_alive_event_count = alive_count.min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    events_preceeding_labels =  ['DIAG', 'LAB', 'DRUG']
    dead_events, alive_events = utils.get_filter_dead_alive(events, mortality)

    alive_encounters = alive_events[np.any([alive_events['event_id'].str.contains(x) for x in events_preceeding_labels], axis=0)]
    dead_encounters = dead_events[np.any([dead_events['event_id'].str.contains(x) for x in events_preceeding_labels], axis=0)]

    # aggregate data - group rows of df based on the values in one or more columns
    grouped_alive = alive_encounters.groupby('patient_id')
    grouped_dead = dead_encounters.groupby('patient_id')

    # Calculate the unique timestamp count for each group
    alive_counts = grouped_alive['timestamp'].nunique()
    dead_counts = grouped_dead['timestamp'].nunique()

    # compute stats
    avg_dead_encounter_count = dead_counts.mean()
    max_dead_encounter_count = dead_counts.max()
    min_dead_encounter_count = dead_counts.min()
    avg_alive_encounter_count = alive_counts.mean()
    max_alive_encounter_count = alive_counts.max()
    min_alive_encounter_count = alive_counts.min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    dead_events, alive_events = utils.get_filter_dead_alive(events, mortality)
    
    # aggregate data - group rows of df based on the values in one or more columns
    grouped_alive = alive_events.groupby('patient_id')
    grouped_dead = dead_events.groupby('patient_id')

    alive_length = grouped_alive.apply(first_last_events_duration)
    dead_length = grouped_dead.apply(first_last_events_duration)

    avg_dead_rec_len = dead_length.mean()
    max_dead_rec_len = dead_length.max()
    min_dead_rec_len = dead_length.min()
    avg_alive_rec_len = alive_length.mean()
    max_alive_rec_len = alive_length.max()
    min_alive_rec_len = alive_length.min()

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def first_last_events_duration(grouped_df):
    """
    TODO: Duration (in number of days) between the first event and last event
    for a given patient.
    """
    # Convert timestamps to datetime objects
    timestamps = pd.to_datetime(grouped_df['timestamp'])
    # Calculate time length as the difference between the maximum and minimum timestamps
    duration = (timestamps.max() - timestamps.min()).days
    return duration

def main():
    train_path = '../data/train/'
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)
    
    
if __name__ == "__main__":
    main()

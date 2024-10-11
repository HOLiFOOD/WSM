import pandas as pd
from scipy.stats import gmean
from utils.time_utils import time_difference, group_by_time_unit

def dov_calc(doc_info, topic_info, time_unit, tw=0.05):
    # Split the data based on the selected time unit
    paper_split = dict(tuple(doc_info.groupby('Time')))

    # Get the maximum time value (e.g., year, month, or day)
    max_time = doc_info['Time'].max()

    # Initialize an empty DataFrame
    dov = pd.DataFrame()

    # Create a list to store individual DataFrames for each time group (year, month, or day)
    dfs = []

    # Iterate over each time group's data
    for time_value, group in doc_info.groupby('Time'):
        topic_count = group['Name'].value_counts().reset_index()
        topic_count.columns = ['Name', 'Count']
        topic_count['Time'] = time_value

        # Get min and max time for this group
        min_time = group['Time'].min()
        max_time = group['Time'].max()

        # Calculate the time difference (based on the time unit)
        time_diff = time_difference(max_time, min_time, time_unit)

        # Create a DataFrame for the current time's DoV values
        df = pd.DataFrame(index=[min_time], columns=topic_count['Name'])
        for _, row in topic_count.iterrows():
            topic = row['Name']
            frequency = row['Count']
            # Ensure the time difference is factored in correctly
            df[topic] = (frequency / len(group)) * (1 - tw * time_diff)

        dfs.append(df)

    # Concatenate all the time-based DataFrames into a single DataFrame
    dov = pd.concat(dfs)

    # Compute the geometric mean for each topic across time units
    dov_means = pd.DataFrame(columns=dov.columns)

    for column in dov.columns:
        column_mean = gmean(dov[column].dropna())
        dov_means[column] = [column_mean * 1000]

    t_DoV = dov_means.melt(var_name="Name", value_name="dov")

    # Merge with topic information
    wsm_dov = pd.merge(t_DoV, topic_info, on='Name')

    return wsm_dov

# Now you can call this function with:
# result = dov_calc(doc_info, topic_info, time_unit, tw=0.05)
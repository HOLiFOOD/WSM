import pandas as pd

# Function to group by a specific time unit (Year, Month, Day)
def group_by_time_unit(df, time_unit):
    if time_unit == 'year':
        return df['Date'].dt.year
    elif time_unit == 'month':
        return df['Date'].dt.to_period('M')  # Month as YYYY-MM
    elif time_unit == 'day':
        return df['Date'].dt.date  # Exact day as YYYY-MM-DD
    else:
        raise ValueError("Invalid time_unit. Choose 'year', 'month', or 'day'.")

# Function to calculate time difference based on a time unit
def time_difference(max_time, min_time, time_unit):
    if time_unit == 'year':
        return max_time - min_time  # Directly subtract the years since they are integers
    elif time_unit == 'month':
        return (max_time.year - min_time.year) * 12 + (max_time.month - min_time.month)  # Total months
    elif time_unit == 'day':
        return (max_time - min_time).days  # Difference in days
    else:
        raise ValueError("Invalid time unit for difference. Choose 'year', 'month', or 'day'.")

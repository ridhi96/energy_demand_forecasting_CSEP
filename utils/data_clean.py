import pandas as pd
import numpy as np
import re
from dateutil import parser
from datetime import datetime, timedelta


def clean_date(date_str):
    """
    This functions takes in a date string and returns a standardized date format.
    """
    # remove extra whitespaces and punctuation and underscore
    date_str = re.sub(r'[^\w\s]', '', date_str.strip())
    date_str = re.sub(r'[_]', '', date_str)

    # replace ordinal numbers with regular numbers
    date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)

    # standardize to YYYY-MM-DD
    try:
        date = parser.parse(date_str)
        return date.strftime('%Y-%m-%d')  
    except Exception as e:
        print(f"Exception occurred for {date_str}")

def convert_to_previous_date(date):
    """
    This functions returns the previous date for a given date.
    """
    try: 
        date = pd.to_datetime(date)
        date = date - pd.DateOffset(days=1)
        return date
    except Exception as e:
        print(f"Exception occurred for {date}")

def missing_dates(start, end, df_date):
    """
    This functions finds the missing dates in a dataframe column between two dates.
    Input:
    start (datetime): start datetime object
    end (datetime): end datetime object
    df_date (df): dataframe columns
    """
    # generate a list of all dates within the time period
    all_dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    
    # gonvert the 'date' column of the DataFrame to datetime objects
    df_date = pd.to_datetime(df_date)
    
    # find missing dates by comparing each date in all_dates with df_date individually
    missing_dates = [date.strftime('%Y-%m-%d') for date in all_dates if date not in df_date.to_list()]
    
    return sorted(missing_dates)

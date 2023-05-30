"""
Python file used to prepare data for clustering by :
    - put all non-numeric values to numeric
    - convert date & hours
"""
import pandas as pd

def clean_data(df):
    """
    return csv_file cleaned and prepared 

    """
    col_to_num(df)
    convert_date(df)


def col_to_num(df):
    """
    change string values to numeric values

    """
    for key in df.keys():
        print(f"\nchange values from : {key} to numeric")

        if isinstance(df[key][0], str) and key != "date":
            change_dict = {}
            i = 0
            for element in df[key].unique():
                change_dict[element] = i
                i += 1
                print(f"{element} => {i}")
            
            print("\nchanging data...")
            print(df[key].replace(change_dict))

def convert_date(df):
    """
    convert date & hour format to a usable one

    """
    print(df)

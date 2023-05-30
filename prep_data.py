"""
Python file used to prepare data for clustering by :
    - put all non-numeric values to numeric
    - convert date & hours
"""
import pandas as pd



def clean_data(df_prep):
    """
    return csv_file cleaned and prepared

    """
    df_prep = remove_useless_column(df_prep)
    df_prep = col_to_num(df_prep)
    df_prep = convert_date(df_prep)

    return df_prep


def remove_useless_column(df_prep):
    """
    return a the data.frame with only useful data

    """
    return df_prep.drop(["Unnamed: 0", "Num_Acc", "ville", "age"], axis="columns")


def col_to_num(df_prep):
    """
    change string values to numeric values

    """
    change_dict = {}
    for key in df_prep.keys():
        print(f"\nchange values from : {key} to numeric")

        if isinstance(df_prep[key][0], str) and key != "date" and key != "id_code_insee":
            change_dict[key] = {}
            i = 0
            for element in df_prep[key].unique():
                change_dict[key][element] = i
                i += 1
                print(f"{element} => {i}")

            print("\nchanging data...")
            df_prep[key] = df_prep[key].replace(change_dict[key])
            print("Complete!")
    
    return df_prep


def convert_date(df_prep):
    """
    convert date & hour format to a usable one

    """
    print("\nchanging str dates to datetime...")

    df_prep["date"] = pd.to_datetime(df_prep["date"])
    df_prep["jour"] = pd.to_datetime(df_prep["date"]).dt.strftime("%d/%m")
    df_prep["heure"] = pd.to_datetime(df_prep["date"]).dt.strftime("%H:%M")
    df_prep = df_prep.drop("date", axis="columns")

    df_prep["an_nais"] = pd.to_datetime(
        df_prep["an_nais"], format="%Y", yearfirst=True
    ).dt.strftime("%Y")
    return df_prep

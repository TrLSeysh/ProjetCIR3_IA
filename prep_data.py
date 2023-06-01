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
    df_prep = invert_lat_lon(df_prep)
    df_prep = change_data(df_prep)

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
    change_dict = pd.DataFrame()

    for key in df_prep.keys():
        print(f"\nchange values from : {key} to numeric")

        if (
            isinstance(df_prep[key][0], str)
            and key != "date"
            and key != "id_code_insee"
        ):
            labels, levels = pd.factorize(df_prep[key])

            for i in enumerate(levels):
                change_dict = pd.concat(
                    [
                        change_dict,
                        pd.DataFrame(
                            data={"index": i[0], "value": i[1]},
                            columns=["index", "value"],
                            index=[key],
                        ),
                    ]
                )

            print("\nchanging data...")
            df_prep[key] = labels
            print("Complete!")
    print(change_dict)
    change_dict.to_excel("conversion_data_to_num.xlsx")
    return df_prep


def convert_date(df_prep):
    """
    convert date & hour format to a usable one

    """
    print("\nchanging str dates to datetime...")

    df_prep["mois"] = pd.to_datetime(df_prep["date"]).dt.month
    df_prep["mois"].info()
    df_prep["heure"] = pd.to_datetime(df_prep["date"]).dt.hour
    df_prep = df_prep.drop("date", axis="columns")

    df_prep["an_nais"] = pd.to_datetime(
        df_prep["an_nais"], format="%Y", yearfirst=True
    ).dt.strftime("%Y")
    return df_prep


def invert_lat_lon(df_prep):
    """
    invert lat & lon for DOM-TOM regions in France

    """
    for element in enumerate(df_prep["id_code_insee"]):
        if element[1][:2] == "97":
            temp_var = df_prep.loc[element[0], "latitude"]
            df_prep.loc[element[0], "latitude"] = df_prep.loc[element[0], "longitude"]
            df_prep.loc[element[0], "longitude"] = temp_var

        if element[1][:2] == "2A":
            temp_insee = df_prep.loc[element[0], "id_code_insee"]
            df_prep.loc[element[0], "id_code_insee"] = "98" + temp_insee[2:]

        if element[1][:2] == "2B":
            temp_insee = df_prep.loc[element[0], "id_code_insee"]
            df_prep.loc[element[0], "id_code_insee"] = "99" + temp_insee[2:]

    return df_prep


def change_data(df_prep):
    """

    Change descr_grav to 2 labels only and remove [lat : 0, lon : 0] point

    """
    df_prep = df_prep.loc[(df_prep["latitude"] != 0) | (df_prep["longitude"] != 0)]

    # 0 : Indemne / 1 : Blessé léger / 2 Tué & blessé grave
    df_prep.loc[:, "descr_grav"] = df_prep["descr_grav"].replace(
        {1: 0, 2: 2, 3: 2, 4: 1}
    )

    return df_prep

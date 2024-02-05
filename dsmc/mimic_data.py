import glob
import os
import shutil
import pandas as pd
import numpy as np
from utils import interpolate_data


def create_clinical_data(
    file_admission, file_patient, file_chart_items, file_chart_events, folder_save
):
    df_columns_lab = [
        "Anion gap",
        "Bicarbonate",
        "Bilirubin",
        "Creatinine",
        "Chloride",
        "Glucose",
        "Hematocrit",
        "Hemoglobin",
        "Lactate",
        "Platelet",
        "Potassium",
        "PT",
        "Sodium",
        "BUN",
        "WBC",
    ]
    df_columns_vital = [
        "Heart Rate",
        "Arterial BP [Systolic]",
        "Arterial BP [Diastolic]",
        "Respiratory Rate",
        "Skin [Temperature]",
        "SpO2",
        "GCS Total",
    ]
    mean_lab_values = np.array(
        [
            13.35,
            25.65,
            3.36,
            1.5,
            104.0,
            134.0,
            29.96,
            10.09,
            2.44,
            235.05,
            4.08,
            17.76,
            138.84,
            29.85,
            11.23,
        ]
    )
    mean_lab_values = np.expand_dims(mean_lab_values, axis=0)
    # create a dataframe with column names df_columns_lab and values mean_lab_values
    df_mean_lab_values = pd.DataFrame(mean_lab_values, columns=df_columns_lab)
    admission_df = pd.read_csv(file_admission, header=0, sep=",")
    admission_df = admission_df.drop(
        columns=[
            string.upper()
            for string in [
                "row_id",
                "dischtime",
                "admission_type",
                "admission_location",
                "discharge_location",
                "insurance",
                "language",
                "religion",
                "marital_status",
                "ethnicity",
                "edregtime",
                "edouttime",
                "hospital_expire_flag",
                "has_chartevents_data",
            ]
        ]
    )
    # keep only the rows that have any value in column 'deathtime' and the 'SEPSIS' value in 'diagnosis' column
    admission_df = admission_df[
        admission_df["DEATHTIME"].notna()
        & admission_df["DIAGNOSIS"].str.contains("SEPSIS")
    ]
    # convert date of format (year-month-day) to days
    admission_df["Mortality"] = (
        pd.to_datetime(admission_df["DEATHTIME"])
        - pd.to_datetime(admission_df["ADMITTIME"])
    ).apply(lambda x: x.total_seconds() / 3600)
    # admission_df['mortality'] = (pd.to_datetime(admission_df['deathtime'].str.split(' ').str[0]) - pd.to_datetime(admission_df['admittime'].str.split(' ').str[0])).dt.days
    admission_df_full = admission_df.copy()
    admission_df = admission_df.drop(columns=["ADMITTIME", "DEATHTIME", "DIAGNOSIS"])
    # sort admission_df by 'subject_id' column
    admission_df = admission_df.sort_values(by=["SUBJECT_ID"])
    admission_df_full = admission_df_full.sort_values(by=["SUBJECT_ID"])

    patient_df = pd.read_csv(file_patient, header=0, sep=",")
    patient_df = patient_df.drop(columns=["ROW_ID", "DOD", "DOD_SSN", "EXPIRE_FLAG"])
    # keep only the rows that have the same values in 'subject_id' column in both dataframes
    patient_df = patient_df[patient_df["SUBJECT_ID"].isin(admission_df["SUBJECT_ID"])]

    patient_df["AGE"] = patient_df["DOD_HOSP"].str.split(r"-").str[0].values.astype(
        int
    ) - patient_df["DOB"].str.split(r"-").str[0].values.astype(int)
    patient_df["AGE"] = patient_df["AGE"].apply(lambda x: 100 if x == 300 else x)
    patient_df = patient_df.drop(columns=["DOB", "DOD_HOSP"])
    # convert 'gender' column to 0 for 'F' and 1 for 'M' in patient_df
    patient_df["GENDER"] = patient_df["GENDER"].map({"F": 0, "M": 1})

    # sort patient_df by 'subject_id' column
    patient_df = patient_df.sort_values(by=["SUBJECT_ID"])

    df_chart_items = pd.read_csv(file_chart_items, header=0, sep=",")
    # keep only the values of 'itemid' column that contain the df_columns_chart values in the 'label' column
    df_chart_items = df_chart_items.drop(
        columns=[
            string.upper()
            for string in [
                "row_id",
                "abbreviation",
                "dbsource",
                "linksto",
                "category",
                "unitname",
                "param_type",
                "conceptid",
            ]
        ]
    )
    df_chart_items = df_chart_items[
        df_chart_items["LABEL"].isin(df_columns_vital + df_columns_lab)
    ]

    df_chart_events = pd.read_csv(
        file_chart_events,
        header=0,
        sep=",",
        usecols=["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
    )
    # keep only the rows that have the same values in 'subject_id' and 'itemid' columns in both dataframes
    df_chart_events = df_chart_events[
        df_chart_events["SUBJECT_ID"].isin(admission_df["SUBJECT_ID"])
        & df_chart_events["ITEMID"].isin(df_chart_items["ITEMID"])
        & df_chart_events["HADM_ID"].isin(admission_df["HADM_ID"])
    ]

    # sort df_chart_events by 'subject_id' and 'itemid' columns
    df_chart_events = df_chart_events.sort_values(by=["SUBJECT_ID", "ITEMID"])

    # replace every Nan value in 'VALUENUM' column with the mean of the previous and next non Nan value
    df_chart_events["VALUENUM"] = df_chart_events["VALUENUM"].fillna(
        df_chart_events["VALUENUM"].interpolate()
    )

    for subject_id in df_chart_events["SUBJECT_ID"].unique():
        # return the sub frame of df_chart_events that contains only the rows with the same value in 'subject_id' column
        itemids = df_chart_events.loc[
            df_chart_events["SUBJECT_ID"] == subject_id, "ITEMID"
        ]
        # print("item ids:", df_chart_events.loc[df_chart_events['SUBJECT_ID'] == subject_id])
        for itemid in itemids.unique():
            time_chart = pd.to_datetime(
                df_chart_events.loc[
                    (df_chart_events["SUBJECT_ID"] == subject_id)
                    & (df_chart_events["ITEMID"] == itemid),
                    "CHARTTIME",
                ]
            )
            # sort the time_chart values
            time_chart = time_chart.sort_values()
            ref_date = time_chart.iloc[0]
            df_chart_events.loc[
                (df_chart_events["SUBJECT_ID"] == subject_id)
                & (df_chart_events["ITEMID"] == itemid),
                "CHARTTIME",
            ] = (time_chart - ref_date).apply(lambda x: x.total_seconds() / 3600)

    df_chart_events = df_chart_events.drop("HADM_ID", axis=1)
    # replace the values of 'item_id' column with the values of 'label' column from df_lab_items after finding the corresponding 'label' value for each 'item_id' value
    id_label_dict = dict(zip(df_chart_items["ITEMID"], df_chart_items["LABEL"]))
    df_chart_events["ITEMID"] = df_chart_events["ITEMID"].map(id_label_dict)

    # remove null values
    df_chart_events = df_chart_events.dropna()

    # convert the 'charttime' column to int values
    df_chart_events["CHARTTIME"] = np.ceil(df_chart_events["CHARTTIME"].values)
    df_demographic = pd.DataFrame(columns=["Age", "Gender", "Mortality"])
    df_lab_all = pd.DataFrame(columns=df_columns_lab)

    # keep admission_df and patient_df rows that exist in df_chart_events based on 'SUBJECT_ID' column
    admission_df = admission_df[
        admission_df["SUBJECT_ID"].isin(df_chart_events["SUBJECT_ID"])
    ]
    patient_df = patient_df[
        patient_df["SUBJECT_ID"].isin(df_chart_events["SUBJECT_ID"])
    ]
    saved_times = 0
    if not os.path.exists(folder_save + "vital_signs"):
        os.makedirs(folder_save + "vital_signs")
    for i, subject_id in enumerate(df_chart_events["SUBJECT_ID"].unique()):
        subset = df_chart_events[df_chart_events["SUBJECT_ID"] == subject_id]
        max_len = 0
        data_list = []
        continue_outer = False
        for column_name in df_columns_vital:
            sub_subset = subset[subset["ITEMID"] == column_name]
            sub_subset = sub_subset[["CHARTTIME", "VALUENUM"]]
            sub_subset = sub_subset.drop_duplicates(subset="CHARTTIME", keep="last")
            sub_subset = sub_subset.sort_values(by=["CHARTTIME"])
            # if there is only one or None value for the current column_name, skip this subject
            if len(sub_subset.values) <= 1:
                continue_outer = True
                break
            cur_len = int(sub_subset["CHARTTIME"].values[-1])
            sub_subset = sub_subset.drop("CHARTTIME", axis=1)
            data_list.append(sub_subset.values)
            if cur_len > max_len:
                max_len = cur_len

        if max_len < 10 or continue_outer:
            continue
        df_time_series = {df_columns_vital[i]: [] for i in range(len(df_columns_vital))}
        j = 0
        for column_name in df_columns_vital:
            df_time_series[column_name] = interpolate_data(data_list[j], max_len)
            j += 1

        df_time_series = pd.DataFrame.from_dict(df_time_series)
        saved_times += 1
        df_time_series.to_csv(
            folder_save + f"vital_signs/time_series_{saved_times}.csv",
            index=False,
            header=True,
        )

        df_lab = pd.DataFrame(columns=df_columns_lab, index=[saved_times])
        for column_name in df_columns_lab:
            sub_subset = subset[subset["ITEMID"] == column_name]
            sub_subset = sub_subset[["CHARTTIME", "VALUENUM"]]
            sub_subset = sub_subset.sort_values(by=["CHARTTIME"])
            if sub_subset.empty:
                df_lab[column_name].values[0] = df_mean_lab_values[column_name].values[
                    0
                ]
            else:
                df_lab[column_name] = sub_subset["VALUENUM"].values[0]
        df_lab_all = pd.concat((df_lab_all, df_lab), axis=0)

        age = patient_df.loc[patient_df["SUBJECT_ID"] == subject_id, "AGE"].values[0]
        mortality = admission_df.loc[
            admission_df["SUBJECT_ID"] == subject_id, "Mortality"
        ].values[0]
        gender = patient_df.loc[
            patient_df["SUBJECT_ID"] == subject_id, "GENDER"
        ].values[0]
        # append the values above to the df_demographic dataframe
        df_demographic = pd.concat(
            (
                df_demographic,
                pd.DataFrame(
                    {"Age": age, "Gender": gender, "Mortality": int(mortality)},
                    index=[saved_times],
                ),
            ),
            axis=0,
        )
    df_demographic.to_csv(folder_save + "demographic.csv", index=False, header=True)
    df_lab_all.to_csv(folder_save + "lab.csv", index=False, header=True)

    # sort and resave the csv files
    csv_files = glob.glob(folder_save + "vital_signs/*.csv")
    df_demographic = pd.read_csv(folder_save + "demographic.csv", header=0)
    df_lab = pd.read_csv(folder_save + "lab.csv", header=0)
    df_lab["Mortality"] = df_demographic["Mortality"]
    lengths = []
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file, header=0)
        lengths.append(len(df))
    # sort the csv files based on the length of the dataframe
    csv_files = [x for _, x in sorted(zip(lengths, csv_files))]

    if not os.path.exists(folder_save + "vital_signs_sorted"):
        os.makedirs(folder_save + "vital_signs_sorted")
    # save again the csv files
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file, header=0)
        df.to_csv(folder_save + f"vital_signs_sorted/time_series_{i}.csv", index=False)
    # delete the content of the old folder and move the sorted files to the old folder
    shutil.rmtree(folder_save + "vital_signs")
    os.mkdir(folder_save + "vital_signs")
    csv_files = glob.glob(folder_save + "vital_signs_sorted/*.csv")
    for file in csv_files:
        shutil.move(file, folder_save + "vital_signs")
    # remove the sorted folder
    shutil.rmtree(folder_save + "vital_signs_sorted")
    # sort the demographic file
    df_demographic = df_demographic.sort_values(by=["Mortality"])
    df_lab = df_lab.sort_values(by=["Mortality"])
    df_lab = df_lab.drop("Mortality", axis=1)

    df_demographic.to_csv(folder_save + "demographic.csv", index=False)
    df_lab.to_csv(folder_save + "lab.csv", index=False)

    # free space in RAM
    del (
        df_demographic,
        df_lab,
        df_lab_all,
        df_time_series,
        df_columns_lab,
        df_columns_vital,
        df_mean_lab_values,
        df_chart_events,
        admission_df,
        patient_df,
    )

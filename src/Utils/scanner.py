import os
import pandas as pd
import numpy as np

def read_beijing_dataset():
    """
    Reads the Beijing FoG dataset (500 hz) from the specified directory structure.
    Returns a dictionary where keys are patient numbers and values are dictionaries of trails.
    Each trail contains a DataFrame with the relevant data.
    """
    base_dir = "../Beijing Dataset/r8gmbtv7w2-FIltered/Filtered Data/"
    patients = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

    dataframes = []

    for patient_no in patients:
        patient_path = os.path.join(base_dir, patient_no)
        trails = sorted([f for f in os.listdir(patient_path) if f.endswith('.txt')])
        patient_df = []
        for trail_no in trails:
            trail_path = os.path.join(patient_path, trail_no)
            df = pd.read_csv(
                trail_path,
                sep=None,
                engine='python',
                header=None,
                usecols=[1, 32, 33, 34, 35, 36, 37, 60],
                names=[
                    'Timestamp',
                    'Acc_x_left', 'Acc_y_left', 'Acc_z_left',
                    'Gyro_x_left', 'Gyro_y_left', 'Gyro_z_left',
                    'FoGClass'
                ]
            )
            patient_df.append(df)
        dataframes.append(patient_df)
    
    feature_columns = ['Acc_x_left', 'Acc_y_left', 'Acc_z_left',
                   'Gyro_x_left', 'Gyro_y_left', 'Gyro_z_left']
    label_column = ['FoGClass']
    sample_rate = 500 
    return dataframes, feature_columns, label_column, sample_rate 

def read_daphnet_dataset():
    """
    Reads all txt files in Daphnet/dataset_fog_release/dataset/.
    Extracts columns: 0 (Time), 1-3 (Ankle acc), and last column (Annotation).
    Returns a dict: {filename: DataFrame}
    """
    base_dir = "../Daphnet/dataset_fog_release/dataset/"
    files = [f for f in os.listdir(base_dir) if f.endswith('.txt')]
    dataframes = []

    for fname in files:
        patient_df = []
        fpath = os.path.join(base_dir, fname)
        # Read first 4 cols and last col (annotation)
        df = pd.read_csv(
            fpath,
            sep=r'\s+',
            header=None,
            usecols=[0, 1, 2, 3, 10],
            names=[
                'Timestamp',
                'Acc_x_left', 'Acc_y_left', 'Acc_z_left',
                'FoGClass'
            ]
        )

        # Remove leading and trailing rows where FoGClass == 0, but keep internal 0s
        mask = df['FoGClass'] != 0

        if mask.any():
            first = mask.idxmax()
            last = 1 + mask[::-1].idxmax()
            df = df.iloc[first:last].reset_index(drop=True)

        # df has multiple trails, each trail is separated by rows with FoGClass == 0, append each trail to patient_df
        patient_df = []
        zero_indices = df.index[df['FoGClass'] == 0].tolist()
        # Add start and end for easier slicing
        split_points = [ -1 ] + zero_indices + [ len(df) ]
        # Convert FoGClass label 1 to 0 and 2 to 1
        df['FoGClass'] = df['FoGClass'].replace({1: 0, 2: 1})

        for i in range(len(split_points) - 1):
            start = split_points[i] + 1
            end = split_points[i+1]
            trail = df.iloc[start:end].reset_index(drop=True)
            if not trail.empty:
                
                patient_df.append(trail)

        dataframes.append(patient_df)

    feature_columns = ['Acc_x_left', 'Acc_y_left', 'Acc_z_left']
    label_column = ['FoGClass']
    sample_rate = 64 
    return dataframes, feature_columns, label_column, sample_rate 



def read_act_dataBase(main_directory = '../ACT data', subfolders = ['left_csv']):  
    # List to hold dataframes
    dataframes = []
    file_names = []
    
    # Loop through each subdirectory in the main directory
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_directory, subfolder)
        
        # Loop through each patient data folder
        for patient_folder in os.listdir(subfolder_path):
            patient_folder_path = os.path.join(subfolder_path, patient_folder)
            patient_df = []
            
            # Check if it's a directory
            if os.path.isdir(patient_folder_path):
                
                # Loop through each CSV file in the patient folder
                for file in os.listdir(patient_folder_path):
                    if file.endswith('.csv'):
                        file_path = os.path.join(patient_folder_path, file)
                        
                        # Read the CSV file and append to the list
                        tmp_df = pd.read_csv(file_path)
                        tmp_df = tmp_df.loc[:, ~tmp_df.columns.str.contains('^Unnamed')]
                        patient_df.append(tmp_df)
                        file_names.append(file_path)
            dataframes.append(patient_df)
            
    feature_columns = ['Gyro_x_left', 'Gyro_y_left', 'Gyro_z_left',
                       'Acc_x_left', 'Acc_y_left', 'Acc_z_left',
                       'Mag_x_left', 'Mag_y_left', 'Mag_z_left']
    label_column = ['FoGClass']
    sample_rate = 50
    return dataframes, feature_columns, label_column, sample_rate 

def flatten_dfs(dataframes):
    # Flatten the nested list using list comprehension
    flat_dataframes = [df for sublist in dataframes for df in sublist]
    return flat_dataframes

def concat_dfs(dataframes):
    # Concatenate all the DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df
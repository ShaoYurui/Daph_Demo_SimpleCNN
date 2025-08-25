from copy import deepcopy
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

class Pipeline:
    def __init__(self):
        self.steps = []
    
    def add_step(self, func):
        """Add a step to the pipeline."""
        self.steps.append(func)

    def execute(self, data, cols_to_process):
        """Execute the pipeline on the input data."""
        for step in self.steps:
            data = step(deepcopy(data), cols_to_process)
        return data
    
class PreProcessor:
    def __init__(self, pre_window_sec=5, sample_rate=50):
        self.pre_window_sec = pre_window_sec
        self.sample_rate = sample_rate
        self.target_rate = 50 #hz
        self._fit_to_normalize = None
        self._fit_to_transform = None 

    def reset(self):
        self._fit_to_normalize = None
        self._fit_to_transform = None 

    def normalize(self, dfs, columns_to_normalize):
        if self._fit_to_normalize is None:
            self._fit_to_normalize = pd.concat(dfs, ignore_index=True)
            # print("Initialising PreProcessor._fit_to_normalize")

        # Combine all dataframes into a single dataframe
        ref_df = self._fit_to_normalize.copy()

        # Compute normalization parameters from the combined DataFrame
        means = ref_df[columns_to_normalize].mean()
        stds = ref_df[columns_to_normalize].std()

        # Normalize each DataFrame in the list
        normalized_dfs = []
        for df in dfs:
            normalized_df = df.copy()
            normalized_df[columns_to_normalize] = (df[columns_to_normalize] - means) / stds
            normalized_dfs.append(normalized_df)
        
        return normalized_dfs
    
    def generate_pre_fog_label(self,  dfs, columns_to_process):
        for df in dfs:
            # Change all FoGClass values from 1 to 2
            df.loc[df['FoGClass'] == 1, 'FoGClass'] = 2
            
            # Find the indices where FoGClass changes from 0 to 1 (start of FoG episode)
            fog_starts = df.index[(df['FoGClass'].shift(1) == 0) & (df['FoGClass'] == 2)].tolist()
            
            # Iterate through each FoG start index
            for start_idx in fog_starts:
                # Calculate the start index for the pre_len rows before the FoG episode
                pre_fog_start = max(0, start_idx - self.pre_window_sec * self.target_rate)

                # Label the pre_len rows before the FoG episode as 1, but only if the original value is 0
                df.loc[pre_fog_start:start_idx-1, 'FoGClass'] = df.loc[pre_fog_start:start_idx-1, 'FoGClass'].apply(lambda x: 1 if x == 0 else x)

        return dfs
    
    def slice_windows(self, dfs, window_size, stride, feature_columns, target_cols):
        input_windows = []
        target_windows = []
        for df in dfs:  # Process each dataframe in the list
            n_samples = df.shape[0]
            for start in range(0, n_samples - window_size * self.target_rate, int(stride * self.target_rate)):
                end = start + window_size * self.target_rate
                input_window = df.iloc[start:end].reset_index(drop=True)
                target_window = df.iloc[start:end].reset_index(drop=True)

                input_windows.append(input_window[feature_columns].values)
                target_windows.append(target_window[target_cols].values[-1])

        return np.array(input_windows), np.array(target_windows)

    def downsample_dfs_to_50hz(self, dfs, columns_to_process):
        """
        Downsamples only columns_to_process in each DataFrame in dfs to target_rate Hz using interpolation.
        Other columns are downsampled by taking the nearest value.
        Returns a new list of downsampled DataFrames.
        """
        downsampled_dfs = []
        for df in dfs:
            df = df.copy()
            # Create a time index in seconds
            time_index = pd.to_timedelta(np.arange(len(df)) / self.sample_rate, unit='s')
            df.index = time_index

            # Downsample columns_to_process with interpolation
            interp_cols = df[columns_to_process].resample(f'{int(1000/self.target_rate)}ms').interpolate('linear')

            # Downsample other columns with nearest
            other_cols = [col for col in df.columns if col not in columns_to_process]
            nearest_cols = df[other_cols].resample(f'{int(1000/self.target_rate)}ms').nearest()

            # Combine
            downsampled = pd.concat([interp_cols, nearest_cols], axis=1)
            downsampled = downsampled[df.columns]  # preserve original column order
            downsampled = downsampled.reset_index(drop=True)
            downsampled_dfs.append(downsampled)
        return downsampled_dfs

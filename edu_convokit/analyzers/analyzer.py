from edu_convokit import utils
from typing import List, Dict, Tuple, Union
import pandas as pd
import os

class Analyzer:
    def __init__(
            self, 
            data_dir: str = None,
            filenames: Union[List[str], str] = None,
            dfs: Union[List[pd.DataFrame], pd.DataFrame] = None,
            max_transcripts: int = None
            ):
        """
        Initialize an analyzer.

        There are several ways to initialize an analyzer:
        - You can specify a data directory, in which case all files in the directory will be loaded.
        - You can specify a list of filenames, in which case all files will be loaded.
        - You can specify a single filename, in which case only that file will be loaded.
        - You can specify a dataframe, in which case the dataframe will be used.

        Note: All of these filenames must be valid analysis files (i.e. .csv, .xlsx, or .json).

        Arguments:
            data_dir (str): directory containing data files
            filenames (Union[List[str], str]): list of filenames to load
            dfs (Union[List[pd.DataFrame], pd.DataFrame]): list of dataframes to load
            max_transcripts (int): maximum number of transcripts to load
        """
        self.data_dir = data_dir
        self.filenames = filenames
        
        self.dfs = dfs 
        if isinstance(self.dfs, pd.DataFrame):
            self.dfs = [self.dfs]

        self.max_transcripts = max_transcripts

        # Load the data into a list of files and df
        if self.data_dir is not None:
            self.filenames = utils.get_valid_analysis_files_in_dir(self.data_dir)
            self.dfs = [utils.load_data(fname) for fname in self.filenames]

            if self.max_transcripts is not None:
                self.filenames = self.filenames[:self.max_transcripts]

            self._df = utils.merge_dataframes_in_list(filenames=self.filenames)
        elif filenames is not None:
            if isinstance(filenames, str):
                self.filenames = [filenames]
            self.dfs = [utils.load_data(fname) for fname in self.filenames]
            self._df = utils.merge_dataframes_in_list(filenames=self.filenames)

    def get_df(self):
        return self._df

    def plot_statistics(self):
        raise NotImplementedError
    
    def report_statistics(self):
        raise NotImplementedError
    
    def print_statistics(self):
        raise NotImplementedError
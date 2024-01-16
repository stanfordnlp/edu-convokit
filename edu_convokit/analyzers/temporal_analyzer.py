from typing import Tuple, List, Dict, Union
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tqdm
from edu_convokit import utils
           
from edu_convokit.analyzers import analyzer

class TemporalAnalyzer(analyzer.Analyzer): 

    def _compute_statistics(
            self,
            speaker_column: str,
            feature_column: str,
            dfs: Union[List[pd.DataFrame], pd.DataFrame] = None,
            num_bins: int = 10,
            dropna: bool = False,
            # additional_columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        Compute statistics for a feature across all speakers.

        Arguments:
            speaker_column: name of column containing speaker names
            feature_column: name of column containing feature to compute statistics for
        """

        if dfs is None:
            dfs = self.dfs
        
        if isinstance(dfs, pd.DataFrame):
            dfs = [dfs]

        results_df = []

        for num_df, df in enumerate(dfs):
            assert speaker_column in df.columns, f"Speaker column {speaker_column} not found in dataframe."
            assert feature_column in df.columns, f"Feature column {feature_column} not found in dataframe."

            # Split the session into chunks
            df_chunks = utils.split_dataframe(df=df, num_bins=num_bins)

            for chunk_i, df_chunk in enumerate(df_chunks):
                if dropna:
                    df_chunk = df_chunk.dropna(subset=[feature_column])

                feature_sum = df_chunk[feature_column].sum()
                for speaker in df_chunk[speaker_column].unique():
                    speaker_df = df_chunk[df_chunk[speaker_column] == speaker]
                    result = {
                        "speaker": speaker,
                        "chunk": chunk_i,
                        f"raw_speaker_{feature_column}": speaker_df[feature_column].sum(),
                        f"avg_speaker_{feature_column}": speaker_df[feature_column].mean(),
                        f"prop_speaker_{feature_column}": speaker_df[feature_column].sum() / feature_sum,
                    }
                    results_df.append(result)

            if self.max_transcripts is not None and num_df >= self.max_transcripts - 1:
                break
        
        results_df = pd.DataFrame(results_df)
        return results_df

    def report_statistics(
            self,
            feature_column: str,
            dfs: Union[List[pd.DataFrame], pd.DataFrame] = None,
            speaker_column: str = None,
            num_bins: int = 10,
            value_as: str = "raw", # raw, avg, prop, all
            dropna: bool = False,
    ) -> str:
        """
        Report statistics for a feature across all speakers across the bins.

        Arguments:
            feature_column (str): name of column containing feature to compute statistics for
            dfs (Union[List[pd.DataFrame], pd.DataFrame]): list of dataframes. If None, use self.dfs from constructor
            speaker_column (str): name of column containing speaker names
            num_bins (int): number of bins to split the data into
            value_as (str): raw, avg, prop, all
            dropna (bool): drop rows with NaN values in feature_column

        Returns:
            str: string representation of statistics
        """
        assert value_as in ["raw", "avg", "prop", "all"], f"Invalid value_as {value_as}. Must be one of ['raw', 'avg', 'prop']."

        results_df = self._compute_statistics(
            speaker_column=speaker_column,
            feature_column=feature_column,
            dropna=dropna,
            num_bins=num_bins,
            dfs=dfs,
        )

        text = ""
        text += f"{feature_column}\n\n"

        for chunk_i in range(num_bins):
            text += f"Chunk {chunk_i}\n\n"

            if value_as == "raw":
                text += "Raw statistics\n"
                text += results_df[results_df["chunk"] == chunk_i].groupby("speaker")[f"raw_speaker_{feature_column}"].describe().to_string()

            elif value_as == "avg":
                text += "Average statistics\n"
                text += results_df[results_df["chunk"] == chunk_i].groupby("speaker")[f"avg_speaker_{feature_column}"].describe().to_string()

            elif value_as == "prop": 
                text += "\n\nProportion statistics\n"
                text += results_df[results_df["chunk"] == chunk_i].groupby("speaker")[f"prop_speaker_{feature_column}"].describe().to_string()

            text += "\n\n"

        return text
    
    def print_statistics(
            self,
            feature_column: str,
            speaker_column: str = None,
            value_as: str = "raw", # raw, avg, prop
            dropna: bool = False,
            dfs: Union[List[pd.DataFrame], pd.DataFrame] = None,
    ):
        """
        Print statistics for a feature across all speakers.

        Arguments:
            feature_column (str): name of column containing feature to compute statistics for
            speaker_column (str): name of column containing speaker names
            value_as (str): raw, avg, prop
            dropna (bool): drop rows with NaN values in feature_column
            dfs (Union[List[pd.DataFrame], pd.DataFrame]): list of dataframes. If None, use self.dfs from constructor

        Returns:
            None
        """
        text = self.report_statistics(
            feature_column=feature_column,
            speaker_column=speaker_column,
            value_as=value_as,
            dropna=dropna, 
            dfs=dfs,
        )
        print(text)


    def plot_temporal_statistics(
            self,
            feature_column: str,
            dfs: Union[List[pd.DataFrame], pd.DataFrame] = None,
            speaker_column: str = None,
            value_as: str = "raw", # raw, avg, prop
            num_bins: int = 10,
            dropna: bool = False,
            title: str = None,
            xlabel: str = None,
            ylabel: str = None,
            save_path: str = None,
            hue: str = None,
            xrange: Tuple[float, float] = None,
            yrange: Tuple[float, float] = None,
            label_mapping: Dict[str, str] = None
        ):
        """
        Plot statistics for a feature across all speakers across bins

        Arguments:
            feature_column (str): name of column containing feature to compute statistics for
            dfs (Union[List[pd.DataFrame], pd.DataFrame]): list of dataframes. If None, use self.dfs from constructor
            speaker_column (str): name of column containing speaker names
            value_as (str): raw, avg, prop
            num_bins (int): number of bins to split the data into
            dropna (bool): drop rows with NaN values in feature_column
            title (str): title of plot
            xlabel (str): x-axis label
            ylabel (str): y-axis label
            save_path (str): path to save plot
            hue (str): name of column to use for hue
            xrange (Tuple[float, float]): x-axis range
            yrange (Tuple[float, float]): y-axis range
            label_mapping (Dict[str, str]): mapping from original label to new label

        Returns:
            None                
        """
        
        sns.set_theme(style="whitegrid")
        sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
        plt.rcParams["font.family"] = "serif"
        
        assert value_as in ["raw", "avg", "prop"], f"Invalid value_as {value_as}. Must be one of ['raw', 'avg', 'prop']."

        results_df = self._compute_statistics(
            speaker_column=speaker_column,
            feature_column=feature_column,
            dropna=dropna,
            num_bins=num_bins,
            dfs=dfs,
        )

        if label_mapping is not None:
            results_df["speaker"] = results_df["speaker"].map(label_mapping)

        y = f"{value_as}_speaker_{feature_column}"

        if hue is None:
            hue = "speaker"

        ax = sns.lineplot(
            x="chunk",
            y=y, 
            data=results_df, 
            hue=hue
        )

        if title is not None:
            ax.set_title(title)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if xrange is not None:
            ax.set_xlim(xrange)

        if yrange is not None:
            ax.set_ylim(yrange)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

        plt.clf()

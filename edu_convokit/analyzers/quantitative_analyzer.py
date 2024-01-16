import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union
import pandas as pd
from edu_convokit.analyzers import analyzer
from edu_convokit import utils
import tqdm
import os

class QuantitativeAnalyzer(analyzer.Analyzer): 

    def _compute_statistics(
            self,
            speaker_column: str,
            feature_column: str,
            df: pd.DataFrame = None,
            dropna: bool = False
    ) -> pd.DataFrame:
        """
        Compute statistics for a feature across all speakers.

        Arguments:
            speaker_column: name of column containing speaker names
            feature_column: name of column containing feature to compute statistics for
        """

        if df is None:
            # Load all the dataframes if no df is passed.
            assert self.filenames is not None, "No filenames passed to analyzer. Either initialize the analyzer with data_dir, filenames, or df, or pass a dataframe to this function."
            dfs = [utils.load_data(fname) for fname in self.filenames]
        else:
            dfs = [df]

        results_df = []

        for df in dfs:
            assert speaker_column in df.columns, f"Speaker column {speaker_column} not found in dataframe." 
            assert feature_column in df.columns, f"Feature column {feature_column} not found in dataframe."

            if dropna:
                df = df.dropna(subset=[feature_column])

            feature_sum = df[feature_column].sum()
            for speaker in df[speaker_column].unique():
                speaker_df = df[df[speaker_column] == speaker]

                result = {
                    "speaker": speaker,
                    f"raw_speaker_{feature_column}": speaker_df[feature_column].sum(),
                    f"avg_speaker_{feature_column}": speaker_df[feature_column].mean(),
                    f"prop_speaker_{feature_column}": speaker_df[feature_column].sum() / feature_sum,
                }

                results_df.append(result)
        
        results_df = pd.DataFrame(results_df)
        return results_df

    def report_statistics(
            self,
            feature_column: str,
            df: pd.DataFrame = None,
            speaker_column: str = None,
            value_as: str = "raw", # raw, avg, prop, all
            dropna: bool = False,
    ) -> str:
        """
        Report statistics for a feature across all speakers.

        Arguments:
            feature_column (str): name of column containing feature to compute statistics for
            df (pd.DataFrame): pandas dataframe. If None, then use self.dfs from constructor
            speaker_column (str): name of column containing speaker names
            value_as (str): raw, avg, prop, all
            dropna (bool): drop rows with NaN values in feature_column

        Returns:
            str: string representation of statistics
        """
    

        assert value_as in ["raw", "avg", "prop", "all"], f"Invalid value_as {value_as}. Must be one of ['raw', 'avg', 'prop']."

        results_df = self._compute_statistics(
            speaker_column=speaker_column,
            feature_column=feature_column,
            df=df,
            dropna=dropna
        )

        text = ""
        text += f"{feature_column}\n\n"

        if value_as == "raw":
            text += "Raw statistics\n"
            text += results_df.groupby("speaker")[f"raw_speaker_{feature_column}"].describe().to_string()
            text += "\n\n"

        elif value_as == "avg":
            text += "Average statistics\n"
            text += results_df.groupby("speaker")[f"avg_speaker_{feature_column}"].describe().to_string()
            text += "\n\n"
        
        elif value_as == "prop":
            text += "Proportion statistics\n"
            text += results_df.groupby("speaker")[f"prop_speaker_{feature_column}"].describe().to_string()

        return text

    
    def print_statistics(
            self,
            feature_column: str,
            df: pd.DataFrame = None,
            speaker_column: str = None,
            value_as: str = "raw", # raw, avg, prop
            dropna: bool = False,
    ):
        """
        Print statistics for a feature across all speakers.

        Arguments:
            feature_column (str): name of column containing feature to compute statistics for
            df (pd.DataFrame): pandas dataframe. If None, then use self.dfs from constructor
            speaker_column (str): name of column containing speaker names
            value_as (str): raw, avg, prop
            dropna (bool): drop rows with NaN values in feature_column

        Returns:
            None
        """
        text = self.report_statistics(
            df=df,
            feature_column=feature_column,
            speaker_column=speaker_column,
            value_as=value_as,
            dropna=dropna
        )
        print(text)

    def plot_statistics(
            self,
            feature_column: str,
            df: pd.DataFrame = None,
            speaker_column: str = None,
            value_as: str = "raw", # raw, avg, prop, all
            dropna: bool = False,
            title: str = None,
            xlabel: str = None,
            ylabel: str = None,
            save_path: str = None,
            xrange: Tuple[float, float] = None,
            yrange: Tuple[float, float] = None,
            label_mapping: Dict[str, str] = None
        ):
        """
        Plot statistics for a feature across all speakers.

        Arguments:
            feature_column (str): name of column containing feature to compute statistics for
            df (pd.DataFrame): pandas dataframe. If None, then use self.dfs from constructor
            speaker_column (str): name of column containing speaker names
            value_as (str): raw, avg, prop, all
            dropna (bool): drop rows with NaN values in feature_column
            title (str): title of plot
            xlabel (str): x-axis label
            ylabel (str): y-axis label
            save_path (str): path to save plot
            xrange (Tuple[float, float]): x-axis range
            yrange (Tuple[float, float]): y-axis range
            label_mapping (Dict[str, str]): mapping from speaker names to labels

        Returns:
            None
        """

        assert value_as in ["raw", "avg", "prop", "all"], f"Invalid value_as {value_as}. Must be one of ['raw', 'avg', 'prop']."
        sns.set_theme(style="whitegrid")
        sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
        plt.rcParams["font.family"] = "serif"

        results_df = self._compute_statistics(
            speaker_column=speaker_column,
            feature_column=feature_column,
            df=df,
            dropna=dropna
        )

        if label_mapping is not None:
            results_df["speaker"] = results_df["speaker"].map(label_mapping)

        y = f"{value_as}_speaker_{feature_column}"
    
        ax = sns.boxplot(
            x="speaker",
            y=y,
            data=results_df
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
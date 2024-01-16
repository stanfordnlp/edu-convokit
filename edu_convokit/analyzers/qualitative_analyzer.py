

from edu_convokit.analyzers import analyzer
from typing import List, Dict, Tuple, Union
import pandas as pd

class QualitativeAnalyzer(analyzer.Analyzer):

    def print_examples(
            self,
            speaker_column: str,
            text_column: str,
            feature_column: str,
            df: pd.DataFrame = None,
            feature_value: Union[str, List[str]] = None, # If None, then use all values
            max_num_values: int = 2,
            max_num_examples: int = 3,
            show_k_previous_lines: int = 0,
            show_k_next_lines: int = 0,
            dropna: bool = False
        ) -> None:
        """
        Get text examples for a feature value.

        Output = 
            [(
            [(speaker, text), ...)], # previous text
            (speaker, current_text), # current text
            [(speaker, text), ...], # next text
            feature_value)
            ), ...]

        Arguments:
            speaker_column (str): name of column containing speaker names
            text_column (str): name of column containing text to get predictions for
            feature_column (str): name of column containing feature to get examples for
            df (pd.DataFrame): pandas dataframe. If None, then use self.dfs from constructor
            feature_value (Union[str, List[str]]): if not None, only get examples for this feature value
            show_k_previous_lines (int): show k previous lines
            show_k_next_lines (int): show k next lines
            dropna (bool): drop rows with NaN values in feature_column

        Returns:
            None
        """        
        examples = self._get_examples(
            df=df,
            speaker_column=speaker_column,
            text_column=text_column,
            feature_column=feature_column,
            feature_value=feature_value,
            max_num_values=max_num_values,
            max_num_examples=max_num_examples,
            show_k_previous_lines=show_k_previous_lines,
            show_k_next_lines=show_k_next_lines,
            dropna=dropna
        )
        print(self._format_examples(examples, feature_column))

    def report_examples(
            self,
            speaker_column: str,
            text_column: str,
            feature_column: str,
            df: pd.DataFrame = None,
            feature_value: Union[float, List[float]] = None, # If None, then use all values
            max_num_values: int = 2,
            max_num_examples: int = 3,
            show_k_previous_lines: int = 0,
            show_k_next_lines: int = 0,
            dropna: bool = False
        ) -> str:
        """
        Get text examples for a feature value.

        Output = 
            [(
            [(speaker, text), ...)], # previous text
            (speaker, current_text), # current text
            [(speaker, text), ...], # next text
            feature_value)
            ), ...]

        Arguments:
            speaker_column (str): name of column containing speaker names
            text_column (str): name of column containing text to get predictions for
            feature_column (str): name of column containing feature to get examples for
            df (pd.DataFrame): pandas dataframe. If None, then use self.dfs from constructor
            feature_value (Union[float, List[float]]): if not None, only get examples for this feature value
            show_k_previous_lines (int): show k previous lines
            show_k_next_lines (int): show k next lines
            dropna (bool): drop rows with NaN values in feature_column

        Returns:
            str: formatted examples
        """        
        examples = self._get_examples(
            df=df,
            speaker_column=speaker_column,
            text_column=text_column,
            feature_column=feature_column,
            feature_value=feature_value,
            max_num_values=max_num_values,
            max_num_examples=max_num_examples,
            show_k_previous_lines=show_k_previous_lines,
            show_k_next_lines=show_k_next_lines,
            dropna=dropna
        )
        return self._format_examples(examples, feature_column)
    
    def _get_examples(
            self,
            speaker_column: str,
            text_column: str,
            feature_column: str,
            df: pd.DataFrame = None,
            feature_value: Union[float, List[float]] = None, # If None, then use all values
            max_num_values: int = 2,
            max_num_examples: int = 3,
            show_k_previous_lines: int = 0,
            show_k_next_lines: int = 0,
            dropna: bool = False
            ) -> List[Tuple[List[Tuple[str, str]], Tuple[str, str], List[Tuple[str, str]], float]]:
        """
        Get text examples for a feature value.

        Output = [(
            [(speaker, text), ...)], # previous text
            (speaker, current_text), # current text
            [(speaker, text), ...], # next text
            feature_value)
        ), ...]

        Arguments:
            speaker_column (str): name of column containing speaker names
            text_column (str): name of column containing text to get predictions for
            feature_column (str): name of column containing feature to get examples for
            df (pd.DataFrame): pandas dataframe. If None, then use self.dfs from constructor
            feature_value (Union[float, List[float]]): if not None, only get examples for this feature value
            show_k_previous_lines (int): show k previous lines
            show_k_next_lines (int): show k next lines
            dropna (bool): drop rows with NaN values in feature_column

        Returns:
            List[Tuple[List[Tuple[str, str]], Tuple[str, str], List[Tuple[str, str]], float]]: list of examples
        """

        if df is None:
            # Merge self.dfs into one df
            df = pd.concat(self.dfs)

        assert text_column in df.columns, f"Text column {text_column} not found in dataframe."
        assert feature_column in df.columns, f"Feature column {feature_column} not found in dataframe."
        
        if dropna:
            df = df.dropna(subset=[feature_column])

        if feature_value is None: 
            feature_value = df[feature_column].unique()

        elif isinstance(feature_value, float): 
            feature_value = [feature_value]

        # Casting types for df and feature_value
        df[text_column] = df[text_column].astype(str)

        examples = []

        num_values = 0
        for value in feature_value:
            # Get all rows with feature_value
            rows = df[df[feature_column] == value]

            
            num_examples = 0
            # Get examples
            for i, row in rows.iterrows():
                # Get previous lines
                prev_lines = []
                for j in range(max(i - show_k_previous_lines, 0), i):
                    if j < 0:
                        continue
                    prev_lines.append((df.iloc[j][speaker_column], df.iloc[j][text_column]))

                # Get next lines
                next_lines = []
                for j in range(i + 1, min(i + show_k_next_lines + 1, len(df))):
                    if j >= len(df):
                        continue
                    next_lines.append((df.iloc[j][speaker_column], df.iloc[j][text_column]))

                num_examples += 1
                examples.append((prev_lines, (row[speaker_column], row[text_column]), next_lines, row[feature_column]))

                if num_examples >= max_num_examples:
                    break
            
            if num_values >= max_num_values:
                break
            num_values += 1
        
        return examples

    def _format_examples(
            self,
            examples: List[Tuple[List[Tuple[str, str]], Tuple[str, str], List[Tuple[str, str]], str]],
            feature_column: str,
            ) -> str:
        """
        Format examples returned by get_examples.

        Output: 

        Feature value: <feature_value>
        <speaker>: <text> # Previous text
        >> <speaker>: <text> # Current text 
        <speaker>: <text> # Next text

        Feature value: <feature_value>
        ...
        """

        formatted_examples = []
        for prev_lines, current_line, next_lines, feature_value in examples:
            formatted_examples.append(f"{feature_column}: {feature_value}")
            for speaker, text in prev_lines:
                formatted_examples.append(f"{speaker}: {text}")
            formatted_examples.append(f">> {current_line[0]}: {current_line[1]}")
            for speaker, text in next_lines:
                formatted_examples.append(f"{speaker}: {text}")
            formatted_examples.append("")

        return "\n".join(formatted_examples)

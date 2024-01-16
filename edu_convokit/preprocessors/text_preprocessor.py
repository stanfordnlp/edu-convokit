import pandas as pd
from typing import List, Union, Tuple
import re
import spacy


class TextPreprocessor:
    def __init__(self):
        pass
    
    def _add_subnames(
            self,
            names: Union[str, List[str]], 
            replacement_names: Union[str, List[str]]
        ) -> Tuple[List[str], List[str]]:
        """
        Add subnames (space-separated) of names to additionally anonymize.
        """

        if isinstance(names, str):
            names = [names]
        if isinstance(replacement_names, str):
            replacement_names = [replacement_names]

        new_names = []
        new_replacement_names = []
        for name, replacement_name in zip(names, replacement_names):
            # Keep original names 
            new_names.append(name)
            new_replacement_names.append(replacement_name)
            name_parts = name.split()
            if len(name_parts) > 1:
                for name_part in name_parts: # Note if name_part is a single letter, it will be anonymized
                    new_names.append(name_part)
                    new_replacement_names.append(replacement_name)

        return new_names, new_replacement_names

    def anonymize_known_names(
            self,
            df: pd.DataFrame,
            text_column: str,
            names: Union[str, List[str]],
            replacement_names: Union[str, List[str]],
            target_text_column: str = None,
            ) -> pd.DataFrame:
        """
        Anonymize a dataframe with known names. 

        Arguments:
            df (pd.DataFrame): pandas dataframe
            text_column (str): name of column containing text to anonymize
            names (Union[str, List[str]]): names to anonymize
            replacement_names (Union[str, List[str]]): replacement names
            target_text_column (str): name of column to store anonymized text. If None, will overwrite text_column.

        Returns:
            pd.DataFrame: dataframe with anonymized text
        """

        assert text_column in df.columns, f"Text column {text_column} not found in dataframe."
        assert len(names) == len(replacement_names), f"Length of names ({len(names)}) must match length of replacement_names ({len(replacement_names)})."

        if target_text_column is None:
            target_text_column = text_column

        # First get subnames (space-separated) of names to additionally anonymize
        names, replacement_names = self._add_subnames(names, replacement_names)

        # Make copy in target_text_column
        df[target_text_column] = df[text_column]

        # Anonymize text while accounting for word boundaries (e.g. "John" should not match "Johnson" or "C" should not match "Charlie")
        for name, replacement_name in zip(names, replacement_names):
            # # Use regular expression to match whole word boundaries
            regex_pattern = r'\b' + re.escape(name) + r'\b'
            df[target_text_column] = df[target_text_column].str.replace(
                pat=regex_pattern,
                repl=replacement_name, 
                regex=True
            )
        
        return df
        
    def anonymize_unknown_names(
            self,
            df: pd.DataFrame,
            text_column: str,
            target_text_column: str = None,
            return_names: bool = False,
            ) -> pd.DataFrame:
        """
        Anonymize a dataframe with unknown names.

        Arguments:
            df (pd.DataFrame): pandas dataframe
            text_column (str): name of column containing text to anonymize
            target_text_column (str): name of column to store anonymized text. If None, will overwrite text_column.
            return_names (bool): if True, return names and replacement_names

        Returns:
            pd.DataFrame: dataframe with anonymized text
            Optional[Tuple[List[str], List[str]]]: names and replacement_names
        """
        nlp = spacy.load("en_core_web_sm")

        assert text_column in df.columns, f"Text column {text_column} not found in dataframe."

        if target_text_column is None:
            target_text_column = text_column

        # Make copy in target_text_column
        df[target_text_column] = df[text_column]

        # Find named entities and create names, replacement_names
        names = []
        for i, row in df.iterrows():
            doc = nlp(row[text_column])
            names.extend([ent.text for ent in doc.ents if ent.label_ == "PERSON" ])
        names = list(set(names)) # Unique names
        replacement_names = [f"[PERSON{i}]" for i in range(len(names))]

        anony_df = self.anonymize_known_names(
            df=df,
            text_column=text_column,
            names=names,
            replacement_names=replacement_names,
            target_text_column=target_text_column
        )

        if return_names:
            return anony_df, (names, replacement_names)

        return anony_df

    def merge_utterances_from_same_speaker(
            self,
            df: pd.DataFrame,
            text_column: str,
            speaker_column: str,
            target_text_column: str,
            ) -> pd.DataFrame:
        """
        Create new dataframe where the utterances from same speaker are grouped together.

        Arguments:
            df (pd.DataFrame): pandas dataframe
            text_column (str): name of column containing text to merge utterances
            speaker_column (str): name of column containing speaker names
            target_text_column (str): name of column to store merged text

        Returns:
            pd.DataFrame: dataframe with merged text
        """
        assert text_column in df.columns, f"Text column {text_column} not found in dataframe."
        assert speaker_column in df.columns, f"Speaker column {speaker_column} not found in dataframe."

        if target_text_column is None:
            target_text_column = text_column

        # Cast text_column to string
        df[text_column] = df[text_column].astype(str)

        new_df = []
        text = ""
        speaker = None

        for i, row in df.iterrows():
            if i == 0:
                text += row[text_column]
                speaker = row[speaker_column]
            else:
                if row[speaker_column] == speaker:
                    text += " " + row[text_column]
                else:
                    new_df.append({
                        target_text_column: text,
                        speaker_column: speaker
                    })
                    text = row[text_column]
                    speaker = row[speaker_column]
        
        # Add last row
        new_df.append({
            target_text_column: text,
            speaker_column: speaker
        })
        return pd.DataFrame(new_df)

    def get_speaker_text_format(
            self,
            df: pd.DataFrame,
            text_column: str,
            speaker_column: str,
            format: str = "{speaker}: {text}",
            ) -> str:
        """
        Return a string with the speaker and text formatted according to the format string.

        Arguments:
            df (pd.DataFrame): pandas dataframe
            text_column (str): name of column containing text
            speaker_column (str): name of column containing speaker names
            format (str): format string

        Returns:
            str: formatted string
        """
        assert text_column in df.columns, f"Text column {text_column} not found in dataframe."
        assert speaker_column in df.columns, f"Speaker column {speaker_column} not found in dataframe."

        text = ""
        for i, row in df.iterrows():
            text += format.format(
                speaker=row[speaker_column], 
                text=row[text_column]) + "\n"

        # Remove last newline
        text = text[:-1]
        return text

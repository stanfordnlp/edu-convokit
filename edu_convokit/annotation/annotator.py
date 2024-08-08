import pandas as pd
from typing import List, Union, Tuple
import spacy
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from edu_convokit import uptake_utils
from scipy.special import softmax
import logging
import re

from edu_convokit.constants import (
    STUDENT_REASONING_HF_MODEL_NAME,
    STUDENT_REASONING_MIN_NUM_WORDS,
    STUDENT_REASONING_MAX_INPUT_LENGTH,
    FOCUSING_HF_MODEL_NAME,
    FOCUSING_MIN_NUM_WORDS,
    FOCUSING_MAX_INPUT_LENGTH,
    UPTAKE_HF_MODEL_NAME,
    UPTAKE_MIN_NUM_WORDS_SPEAKER_A,
    HIGH_UPTAKE_THRESHOLD,
    UPTAKE_MAX_INPUT_LENGTH,
    MATH_PREFIXES,
    MATH_WORDS,
    TEACHER_TALK_MOVES_HF_MODEL_NAME,
    STUDENT_TALK_MOVES_HF_MODEL_NAME
)


class Annotator:
    """
        Annotator class for edu-convokit. Contains methods for annotating data.
    """
    def __init__(self):
        pass

    def _populate_analysis_unit(
            self,
            df: pd.DataFrame,
            analysis_unit: str,
            text_column: str,
            time_start_column: str,
            time_end_column: str,
            output_column: str,
            ) -> pd.DataFrame:
        """
        Populate output_column with number of words, sentences, or timestamps.
        """

        if analysis_unit == "words":
            df[output_column] = df[text_column].str.split().str.len()
        elif analysis_unit == "sentences":
            # Use nlp to split text into sentences
            nlp = spacy.load("en_core_web_sm")
            df[output_column] = df[text_column].apply(lambda x: len(list(nlp(x).sents)))
        elif analysis_unit == "timestamps":
            # Check type of time_start_column and time_end_column
            if df[time_start_column].dtype != "float64":
                df[time_start_column] = df[time_start_column].astype("float64")
            if df[time_end_column].dtype != "float64":
                df[time_end_column] = df[time_end_column].astype("float64")
            df[output_column] = df[time_end_column] - df[time_start_column]
        else:
            raise ValueError(f"Analysis unit {analysis_unit} not supported.")
        return df

    def get_talktime(
            self,
            df: pd.DataFrame,
            text_column: str = None,
            analysis_unit: str = "words", # words, sentences, timestamps
            representation: str = "frequency", # frequency
            time_start_column: str = None,
            time_end_column: str = None,
            output_column: str = "talktime_analysis",
            ) -> pd.DataFrame:
        """
        Analyze talk time of speakers in a dataframe. Return original df and new dataframe with talk time analysis.

        Arguments:
            df (pd.DataFrame): dataframe to analyze
            text_column (str): name of column containing text to analyze. Only required if analysis_unit is words or sentences.
            analysis_unit (str): unit to analyze. Choose from "words", "sentences", "timestamps".
            representation (str): representation of talk time. Choose from "frequency", "proportion".
            time_start_column (str): name of column containing start time. Only required if analysis_unit is timestamps.
            time_end_column (str): name of column containing end time. Only required if analysis_unit is timestamps.
            output_column (str): name of column to store result.

        Returns:
            df (pd.DataFrame): dataframe with talk time analysis
        """ 
        assert analysis_unit in ["words", "sentences", "timestamps"], f"Analysis unit {analysis_unit} not supported."
        assert representation in ["frequency", "proportion"], f"Representation {representation} not supported."

        if text_column is not None and analysis_unit in ["words", "sentences"]:
            assert text_column in df.columns, f"Text column {text_column} not found in dataframe."

        if time_start_column is not None and analysis_unit == "timestamps":
            assert time_start_column in df.columns, f"Time start column {time_start_column} not found in dataframe."
            assert time_end_column in df.columns, f"Time end column {time_end_column} not found in dataframe."

        # First populate output_column with number of words, sentences, or timestamps
        df = self._populate_analysis_unit(df, analysis_unit, text_column, time_start_column, time_end_column, output_column)

        # Return dataframe with talk time analysis
        if representation == 'proportion':
            total = df[output_column].sum()
            df[output_column] = df[output_column] / total

        return df
    
    # HF models
    def _initialize(self, model_shortname):
        # Load model directly
        tokenizer = AutoTokenizer.from_pretrained(model_shortname)
        model = AutoModelForSequenceClassification.from_pretrained(model_shortname)
        model.eval()
        return tokenizer, model

    def _get_classification_predictions(
            self,
            df: pd.DataFrame,
            text_column: str,
            output_column: str,
            model_name: str,
            min_num_words: int = 0,
            max_num_words: int = None,
            speaker_column: str = None,
            speaker_value: Union[str, List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get classification predictions for a dataframe.

        Arguments:
            df: pandas dataframe
            text_column: name of column containing text to get predictions for
            output_column: name of column to store predictions
            speaker_column: name of column that contains speaker names.
            speaker_value: if speaker_column is not None, only get predictions for this speaker.
            model_name: name of model to use.
        """
        assert text_column in df.columns, f"Text column {text_column} not found in dataframe."

        if output_column in df.columns:
            logging.warning(f"Target column {output_column} already exists in dataframe. Skipping.")
            return df

        if speaker_column is not None:
            assert speaker_column in df.columns, f"Speaker column {speaker_column} not found in dataframe."

            if isinstance(speaker_value, str):
                speaker_value = [speaker_value]

        tokenizer, model = self._initialize(model_name)

        # Get predictions
        predictions = []
        for i, row in df.iterrows():
            # Skip if speaker doesn't match
            if speaker_column is not None:
                if row[speaker_column] not in speaker_value:
                    predictions.append(None)
                    continue

            text = row[text_column]

            # Skip if text is too short
            if len(text.split()) < min_num_words:
                predictions.append(None)
                continue

            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_num_words)
                outputs = model(**inputs)
                logits = outputs.logits
                predictions.append(logits.argmax().item())

        df[output_column] = predictions
        return df

    def get_student_reasoning(
            self,
            df: pd.DataFrame,
            text_column: str,
            output_column: str,
            speaker_column: str = None,
            speaker_value: Union[str, List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get student reasoning predictions for a dataframe.

        Arguments:
            df (pd.DataFrame): dataframe to analyze
            text_column (str): name of column containing text to analyze
            output_column (str): name of column to store result
            speaker_column (str): name of column containing speaker names. Only required if speaker_value is not None.
            speaker_value (str or list): if speaker_column is not None, only get predictions for this speaker.
        
        Returns:
            df (pd.DataFrame): dataframe with student reasoning predictions
        """

        # Print out note that the predictions should only be run on student reasoning as that's what the model was trained on.
        logging.warning("""Note: This model was trained on student reasoning, so it should be used on student utterances.
    For more details on the model, see https://arxiv.org/pdf/2211.11772.pdf""")

        return self._get_classification_predictions(
            df=df,
            text_column=text_column,
            output_column=output_column,
            model_name=STUDENT_REASONING_HF_MODEL_NAME,
            min_num_words=STUDENT_REASONING_MIN_NUM_WORDS,
            max_num_words=STUDENT_REASONING_MAX_INPUT_LENGTH,
            speaker_column=speaker_column,
            speaker_value=speaker_value
        )
    
    def get_teacher_talk_moves(
            self, 
            df: pd.DataFrame,
            text_column: str,
            output_column: str,
            speaker_column: str = None,
            speaker_value: Union[str, List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get teacher talk move predictions for a dataframe.

        Arguments:
            df (pd.DataFrame): dataframe to analyze
            text_column (str): name of column containing text to analyze
            output_column (str): name of column to store result
            speaker_column (str): name of column containing speaker names. Only required if speaker_value is not None.
            speaker_value (str or list): if speaker_column is not None, only get predictions for this speaker.

        Returns:
            df (pd.DataFrame): dataframe with teacher talk move predictions
        """

        logging.warning("""Note: This model was trained on teacher talk moves, so it should be used on teacher utterances.
    For more details on the model, see https://github.com/SumnerLab/TalkMoves/tree/main""")

        return self._get_classification_predictions(
            df=df,
            text_column=text_column,
            output_column=output_column,
            model_name=TEACHER_TALK_MOVES_HF_MODEL_NAME,
            min_num_words=0,
            max_num_words=None,
            speaker_column=speaker_column,
            speaker_value=speaker_value
        )
        
    def get_student_talk_moves(
            self, 
            df: pd.DataFrame,
            text_column: str,
            output_column: str,
            speaker_column: str = None,
            speaker_value: Union[str, List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get student talk move predictions for a dataframe.

        Arguments:
            df (pd.DataFrame): dataframe to analyze
            text_column (str): name of column containing text to analyze
            output_column (str): name of column to store result
            speaker_column (str): name of column containing speaker names. Only required if speaker_value is not None.
            speaker_value (str or list): if speaker_column is not None, only get predictions for this speaker.

        Returns:
            df (pd.DataFrame): dataframe with teacher talk move predictions
        """

        logging.warning("""Note: This model was trained on student talk moves, so it should be used on student utterances.
    For more details on the model, see https://github.com/SumnerLab/TalkMoves/tree/main""")

        return self._get_classification_predictions(
            df=df,
            text_column=text_column,
            output_column=output_column,
            model_name=STUDENT_TALK_MOVES_HF_MODEL_NAME,
            min_num_words=0,
            max_num_words=None,
            speaker_column=speaker_column,
            speaker_value=speaker_value
        )

    def get_focusing_questions(
            self,
            df: pd.DataFrame,
            text_column: str,
            output_column: str,
            speaker_column: str = None,
            speaker_value: Union[str, List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get focusing question predictions for a dataframe.

        Arguments:
            df (pd.DataFrame): dataframe to analyze
            text_column (str): name of column containing text to analyze
            output_column (str): name of column to store result
            speaker_column (str): name of column containing speaker names. Only required if speaker_value is not None.
            speaker_value (str or list): if speaker_column is not None, only get predictions for this speaker.

        Returns:
            df (pd.DataFrame): dataframe with focusing question predictions
        """

        logging.warning("""Note: This model was trained on teacher focusing questions, so it should be used on teacher utterances.
    For more details on the model, see https://aclanthology.org/2022.bea-1.27.pdf""")

        return self._get_classification_predictions(
            df=df,
            text_column=text_column,
            output_column=output_column,
            model_name=FOCUSING_HF_MODEL_NAME,
            min_num_words=FOCUSING_MIN_NUM_WORDS,
            max_num_words=FOCUSING_MAX_INPUT_LENGTH,
            speaker_column=speaker_column,
            speaker_value=speaker_value
        )

    def _get_uptake_prediction(self, model, device, instance):
        instance["attention_mask"] = [1] * len(instance["input_ids"])
        for key in ["input_ids", "token_type_ids", "attention_mask"]:
            instance[key] = torch.tensor(instance[key]).unsqueeze(0)  # Batch size = 1
            instance[key] = instance[key].to(device)

        output = model(input_ids=instance["input_ids"],
                        attention_mask=instance["attention_mask"],
                        token_type_ids=instance["token_type_ids"],
                        return_pooler_output=False)
        return output

    def get_uptake(
        self,
        df: pd.DataFrame,
        text_column: str,
        output_column: str,
        speaker_column: str, # Mandatory because we are interested in measuring speaker2's uptake of speaker1's words
        speaker1: Union[str, List[str]], # speaker1 is the student
        speaker2: Union[str, List[str]], # speaker2 is the teacher
        result_type: str = "binary", # raw: uptake score, binary: 1 if uptake score > threshold, 0 otherwise
    ) -> pd.DataFrame:
        """
        Get uptake predictions for a dataframe.
        Following the implementation here:
        https://huggingface.co/ddemszky/uptake-model/blob/main/handler.py

        Arguments:
            df (pd.DataFrame): dataframe to analyze
            text_column (str): name of column containing text to analyze
            output_column (str): name of column to store result
            speaker_column (str): name of column containing speaker names.
            speaker1 (str or list): speaker1 is the student
            speaker2 (str or list): speaker2 is the teacher
            result_type (str): raw or binary

        Returns:
            df (pd.DataFrame): dataframe with uptake predictions
        """

        logging.warning("""Note: This model was trained on teacher's uptake of student's utterances. So, speaker1 should be the student and speaker2 should be the teacher.
    For more details on the model, see https://arxiv.org/pdf/2106.03873.pdf""")

        logging.warning("""Note: It's recommended that you merge utterances from the same speaker before running this model. You can do that with edu_convokit.text_preprocessing.merge_utterances_from_same_speaker.""")

        assert text_column in df.columns, f"Text column {text_column} not found in dataframe."
        assert speaker_column in df.columns, f"Speaker column {speaker_column} not found in dataframe."

        if output_column in df.columns:
            logging.warning(f"Target column {output_column} already exists in dataframe. Skipping.")
            return df

        if isinstance(speaker1, str):
            speaker1 = [speaker1]

        if isinstance(speaker2, str):
            speaker2 = [speaker2]

        # Uptake model is run slightly differently. So this is a separate function.
        input_builder, device, model = uptake_utils._initialize(UPTAKE_HF_MODEL_NAME)

        predictions = []

        with torch.no_grad():
            for i, row in df.iterrows():
                if i == 0:
                    predictions.append(None)
                    continue

                s1 = df[speaker_column].iloc[i-1]
                s2 = df[speaker_column].iloc[i]
                textA = df[text_column].iloc[i-1]
                textB = df[text_column].iloc[i]

                # Skip if text is too short
                if len(textA.split()) < UPTAKE_MIN_NUM_WORDS_SPEAKER_A:
                    predictions.append(None)
                    continue

                if s1 in speaker1 and s2 in speaker2:
                    textA = uptake_utils._get_clean_text(textA, remove_punct=False)
                    textB = uptake_utils._get_clean_text(textB, remove_punct=False)

                    instance = input_builder.build_inputs([textA], textB,
                                                            max_length=UPTAKE_MAX_INPUT_LENGTH,
                                                            input_str=True)
                    output = self._get_uptake_prediction(model, device, instance)
                    uptake_score = softmax(output["nsp_logits"][0].tolist())[1]
                    if result_type == "binary":
                        uptake_score = 1 if uptake_score > HIGH_UPTAKE_THRESHOLD else 0

                    predictions.append(uptake_score)
                else:
                    predictions.append(None)
        df[output_column] = predictions

        return df

    # Math density
    def _load_math_terms(self):
        """
        modify_stem <- function(s) {
                return(paste0('(^|[^a-zA-Z])', s,'(s|es)?([^a-zA-Z]|$)'))
                }

        modify_list <- c('sum', 'arc', 'mass', 'digit', 'graph', 
                        'liter', 'gram', 'add', 'angle', 'scale',
                        'data', 'array', 'ruler', 'meter', 'total',
                        'unit', 'prism', 'median', 'ratio', 'area')

        # Modify those entries in the glossary
        gloss[gloss %in% modify_list] <- modify_stem(gloss[gloss %in% modify_list])

        For every term in MATH_WORDS, we modify the term to include the regex pattern that matches the term and its plural form if it is in MATH_PREFIXES.
        """
        math_terms = []
        for term in MATH_WORDS:
            if term in MATH_PREFIXES:
                math_terms.append(f"(^|[^a-zA-Z]){term}(s|es)?([^a-zA-Z]|$)")
            else:
                math_terms.append(term)
        return math_terms

    def get_math_density(
            self,
            df: pd.DataFrame,
            text_column: str,
            output_column: str,
            count_type: str = "total", # total
            result_type: str = "total", # total, proportion
    ) -> pd.DataFrame:
        """
        Get math density for a dataframe. Following the implementation here: https://edworkingpapers.com/sites/default/files/ai23-855.pdf

        Arguments:
            df (pd.DataFrame): dataframe to analyze
            text_column (str): name of column containing text to analyze
            output_column (str): name of column to store result
            count_type (str): total or unique
            result_type (str): total or proportion

        Returns:
            df (pd.DataFrame): dataframe with math density analysis
        """
        assert text_column in df.columns, f"Text column {text_column} not found in dataframe."
        # assert count_type in ["total", "unique"], f"Count type {count_type} not supported. Choose from 'total' or 'unique'."
        assert result_type in ["total", "proportion"], f"Result type {result_type} not supported. Choose from 'total' or 'proportion'."

        if output_column in df.columns:
            logging.warning(f"Result column {output_column} already exists in dataframe. Skipping.")
            return df

        math_terms = sorted(self._load_math_terms(), key=len, reverse=True)

        df = df.copy()
        df[output_column] = 0

        # Speaker 2 unique terms found
        for i, utt in df.iterrows():
            text = utt[text_column]

            # Count number of math terms in text
            total = 0

            # Check if term is already matched
            matched_positions = set()

            for term in math_terms:
                matches = list(re.finditer(term, text, re.IGNORECASE))
                matches = [match for match in matches if not any(match.start() in range(existing[0], existing[1]) for existing in matched_positions)]
                count = len(matches)
                total += count

                matched_positions.update((match.start(), match.end()) for match in matches)
                
            # Store result
            df.loc[i, output_column] = total

        return df


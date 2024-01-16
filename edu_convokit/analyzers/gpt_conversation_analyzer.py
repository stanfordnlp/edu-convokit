"""
This analyzer is for analyzing full *conversations* using models from OpenAI.
The prompts should be placed under prompts/conversation
"""
import pandas as pd
from typing import List, Union, Tuple
import tqdm
import itertools
import logging
import os
from edu_convokit import utils
from edu_convokit.analyzers import analyzer
from edu_convokit.preprocessors.token_preprocessor import TokenPreprocessor
from edu_convokit.constants import (
    CONVERSATION_PROMPTS_DIR,
    OPENAI_MODEL_2_CONTEXT_LENGTH
)
import openai

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from dotenv import load_dotenv

load_dotenv()

class GPTConversationAnalyzer(analyzer.Analyzer):

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(2))
    def completion_with_backoff(self, prompt, model, temperature=0.0, max_tokens=None):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
                        model=model,
                        messages=[
                                {"role": "user", "content": prompt}
                            ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
        revision = response['choices'][0]['message']['content']
        return revision
    
    def preview_prompt(
        self,
        df: pd.DataFrame,
        prompt_name: str, # Should match the prompt name under CONVERSATION_PROMPTS_DIR
        text_column: str,
        speaker_column: str,
        model: str = "gpt-4",
        add_line_numbers: bool = False,
        format_template: str = "{speaker}: {text}",
        keep_transcript_fraction: float = None,
        ) -> str:
        """
        Preview a prompt on a dataframe and return the prompt.

        Arguments:
            df (pd.DataFrame): pandas dataframe
            prompt_name (str): name of prompt
            text_column (str): name of column containing text
            speaker_column (str): name of column containing speaker names
            model (str): model name
            add_line_numbers (bool): whether to add line numbers
            format_template (str): format string
            keep_transcript_fraction (float): fraction of transcript to keep

        Returns:
            str: prompt
        """
        if not prompt_name.endswith(".txt"):
            prompt_name += ".txt"

        assert text_column in df.columns, f"Text column {text_column} not found in dataframe."
        assert speaker_column in df.columns, f"Speaker column {speaker_column} not found in dataframe."
        assert model in OPENAI_MODEL_2_CONTEXT_LENGTH, f"Model {model} not found in OPENAI_MODEL_2_CONTEXT_LENGTH."

        # Load the prompt.
        prompt_fpath = '/'.join(('prompts', 'conversation', prompt_name))
        prompt_template = utils.load_text_file(prompt_fpath)

        # Determine max context length
        max_context_length = OPENAI_MODEL_2_CONTEXT_LENGTH[model]

        # Determine prompt length
        model_tp = TokenPreprocessor(model=model)
        prompt_length = model_tp.get_num_tokens_from_string(prompt_template)

        # Set budget
        if keep_transcript_fraction is None:
            budget = max_context_length - prompt_length
        else:
            # Get full transcript text
            transcript_txt = "\n".join(df[text_column].tolist())
            # Get transcript length
            budget = int(model_tp.get_num_tokens_from_string(transcript_txt) * keep_transcript_fraction)

        # Format transcript within budget 
        transcript_txt = model_tp.format_transcript_within_budget(
            df=df,
            text_column=text_column,
            speaker_column=speaker_column,
            max_token_budget=budget,
            format_template=format_template,
            add_line_numbers=add_line_numbers,
            print_num_tokens=False,
        )

        # Format prompt
        prompt = prompt_template.format(conversation=transcript_txt)
        return prompt
            
    def run_prompt(
        self,
        df: pd.DataFrame,
        prompt_name: str, # Should match the prompt name under CONVERSATION_PROMPTS_DIR
        text_column: str,
        speaker_column: str,
        model: str = "gpt-4",
        add_line_numbers: bool = False,
        format_template: str = "{speaker}: {text}",
        temperature: float = 0.0,
        max_tokens: int = None,
        keep_transcript_fraction: float = None,
        ) -> Tuple[str, str]:
        """
        Run a prompt on a dataframe and return the (prompt, response) pair.

        Arguments:
            df (pd.DataFrame): pandas dataframe
            prompt_name (str): name of prompt
            text_column (str): name of column containing text
            speaker_column (str): name of column containing speaker names
            model (str): model name
            add_line_numbers (bool): whether to add line numbers
            format_template (str): format string
            temperature (float): temperature
            max_tokens (int): maximum number of tokens
            keep_transcript_fraction (float): fraction of transcript to keep

        Returns:
            Tuple[str, str]: (prompt, response) pair
        """
        prompt = self.preview_prompt(
            df=df,
            prompt_name=prompt_name,
            text_column=text_column,
            speaker_column=speaker_column,
            model=model,
            add_line_numbers=add_line_numbers,
            format_template=format_template,
            keep_transcript_fraction=keep_transcript_fraction
        )

        # Run the prompt
        output = self.completion_with_backoff(
            prompt=prompt, 
            model=model, 
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return prompt, output




import pandas as pd
from typing import Any, List, Union, Tuple
import tiktoken

class TokenPreprocessor: 

    def __init__(self, model: str):
        self.model = model
        pass

    def _format_text_for_token_counting(
            self,
            text: Union[str, List[str], List[dict]],
    ) -> List[dict]:
        """
        Format text for token counting. Final output is a list of dictionaries with keys "role" and "content".
        """
        if isinstance(text, str):
            text = [text]

        if isinstance(text[0], str):
            text = [{"role": "user", "content": text}]
        
        elif isinstance(text[0], dict):
            contains_keys = all([key in _ for _ in text for key in ["role", "content"]])
            assert contains_keys, "Each dictionary in text must contain keys 'role' and 'content'."

        return text
    
    def get_num_tokens_from_string(
            self,
            string: str,
    ) -> int:
        """
        Returns the number of tokens in a text string. From https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb.
        """
        encoding = tiktoken.encoding_for_model(self.model)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def get_num_tokens_from_messages(
            self,
            messages: Union[str, List[str], List[dict]],
    ) -> int:
        """
        Return the number of tokens in a string or list of strings. Code adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        Arguments:
            text (Union[str, List[str]]): string or list of strings

        Returns:
            Union[int, List[int]]: number of tokens in text
        """

        messages = self._format_text_for_token_counting(messages)
        
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        if self.model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        
        elif self.model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4 # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1 # if there's a name, the role is omitted

        elif "gpt-3.5-turbo" in self.model:
            print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            self.model = "gpt-3.5-turbo-0613"
            return self.get_num_tokens_from_messages(messages)
        elif "gpt-4" in self.model:
            print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            self.model = "gpt-4-0613"
            return self.get_num_tokens_from_messages(messages)
        else:
            raise NotImplementedError(
                f"""get_num_tokens_from_messages() is not implemented for model {self.model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def format_transcript_within_budget(
            self, 
            df: pd.DataFrame,
            text_column: str,
            speaker_column: str,
            max_token_budget: int,
            format_template: str = "{speaker}: {text}",
            add_line_numbers: bool = False, # {line} must be in format_template
            print_num_tokens: bool = False,
    ) -> str:
        """
        Format a transcript within a token budget. 

        Arguments:
            df (pd.DataFrame): pandas dataframe
            text_column (str): name of column containing text
            speaker_column (str): name of column containing speaker names
            max_token_budget (int): maximum number of tokens
            format_template (str): format string
            add_line_numbers (bool): whether to add line numbers
            print_num_tokens (bool): whether to print the number of tokens

        Returns:
            str: formatted string
        """
        assert text_column in df.columns, f"Text column {text_column} not found in dataframe."
        assert speaker_column in df.columns, f"Speaker column {speaker_column} not found in dataframe."
        assert "{speaker}" in format_template, "format_template must contain {speaker}."
        assert "{text}" in format_template, "format_template must contain {text}."
        if add_line_numbers:
            assert "{line}" in format_template, "format_template must contain {line}."

        text = ""
        num_tokens = 0
        line = 0
        # Naive approach: add as many rows as possible until max_token_budget is reached
        for i, row in df.iterrows():
            next_text = format_template.format(
                speaker=row[speaker_column], 
                text=row[text_column],
                line=line) + "\n"
            next_text_num_tokens = self.get_num_tokens_from_string(next_text)

            if num_tokens + next_text_num_tokens <= max_token_budget:
                text += next_text
                num_tokens += next_text_num_tokens
                line += 1
            else:
                break

        # Remove last newline
        text = text[:-1]
        if print_num_tokens:
            print(f"Number of tokens: {num_tokens}")
        return text
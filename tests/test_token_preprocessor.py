import unittest
from unittest import TestCase
from edu_convokit.preprocessors.token_preprocessor import TokenPreprocessor
import pandas as pd
import re

speaker_df = pd.DataFrame({
    "text": [
        "Hey.",
        "How are you?",
        "Good, how are you?",
        "I'm good, thanks."
    ],
    "speaker": [
        "A",
        "A",
        "B",
        "A"
    ]
})

models = [
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4",
]

class TestTokenPreprocessor(TestCase):

    def test_get_num_tokens_from_string(self):
        sentence = "tiktoken is great!"

        for model in models:
            tp = TokenPreprocessor(model=model)
            num_tokens = tp.get_num_tokens_from_string(sentence)
            self.assertTrue(num_tokens > 0)

    def test_get_num_tokens_from_messages(self):
        messages = [
            {
                "role": "A",
                "content": "Hey."
            },
            {
                "role": "B",
                "content": "How are you?"
            },
            {
                "role": "A",
                "content": "Good, how are you?"
            },
            {
                "role": "B",
                "content": "I'm good, thanks."
            }
        ]

        for model in models:
            tp = TokenPreprocessor(model=model)
            # Test that it works
            num_tokens = tp.get_num_tokens_from_messages(messages)
            self.assertTrue(num_tokens > 0)
            
            
    def test_format_transcript_within_budget(self):
        for model in models:
            tp = TokenPreprocessor(model=model)
            # Test that it works
            formatted_transcript = tp.format_transcript_within_budget(
                df=speaker_df, 
                text_column="text", 
                speaker_column="speaker", 
                max_token_budget=25,
                format_template="{speaker}: {text}",
                )

            self.assertTrue(len(formatted_transcript) > 0)

    def test_format_transcript_within_budget_with_line_numbers(self):
        for model in models:
            tp = TokenPreprocessor(model=model)
            # Test that it works
            formatted_transcript = tp.format_transcript_within_budget(
                df=speaker_df, 
                text_column="text", 
                speaker_column="speaker", 
                max_token_budget=20,
                format_template="Line {line}. {speaker}: {text}",
                add_line_numbers=True
                )

            self.assertTrue(len(formatted_transcript) > 0)






if __name__ == "__main__":
    unittest.main()
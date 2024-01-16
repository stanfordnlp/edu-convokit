import unittest
from unittest import TestCase
from edu_convokit import utils

from edu_convokit.analyzers import GPTConversationAnalyzer

class TestGPTConversationAnalyzer(TestCase):

    def test_preview_prompt(self):
        analyzer = GPTConversationAnalyzer()
        fpath = "data/ncte/3.csv"
        df = utils.load_data(fpath)
        
        prompt = analyzer.preview_prompt(
            df=df,
            prompt_name="summarize",
            text_column="text",
            speaker_column="speaker",
            model="gpt-4",
            add_line_numbers=True,
            format_template="Line {line}. {speaker}: {text}",
        )

        self.assertTrue(len(prompt) > 0)

    def test_run_prompt(self):
        analyzer = GPTConversationAnalyzer()
        fpath = "data/ncte/3.csv"
        df = utils.load_data(fpath)
        
        prompt, response = analyzer.run_prompt(
            df=df,
            prompt_name="mqi_suggestions",
            text_column="text",
            speaker_column="speaker",
            model="gpt-4",
            add_line_numbers=True,
            format_template="Line {line}. {speaker}: {text}",
        )
        self.assertTrue(len(response) > 0)

    def test_run_prompt_with_fraction(self):
        analyzer = GPTConversationAnalyzer()
        fpath = "data/ncte/3.csv"
        df = utils.load_data(fpath)
        
        prompt, response = analyzer.run_prompt(
            df=df,
            prompt_name="mqi_suggestions",
            text_column="text",
            speaker_column="speaker",
            model="gpt-4",
            add_line_numbers=True,
            format_template="Line {line}. {speaker}: {text}",
            keep_transcript_fraction=0.25,
        )

        self.assertTrue(len(response) > 0)


if __name__ == "__main__":
    unittest.main()
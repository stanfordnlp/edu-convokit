import unittest
from unittest import TestCase
from edu_convokit.preprocessors.text_preprocessor import TextPreprocessor
import pandas as pd
import re

df = pd.DataFrame({
    "text": [
        "Hello, my name is Alice Wang.", 
        "Hello, my name is Bob Clarke.", 
        "Hello, my name is C J.",
        "Hey Johnson, this is John."
    ]
})

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

class TestTextPreprocessing(TestCase):

    def test_anonymize_known_names(self):
        processor = TextPreprocessor()
        names = ["Alice Wang", "Bob Clarke", "C J", "John Paul Jones", "Johnson"]
        replacement_names = ["[TEACHER]", "[STUDENT2]", "[STUDENT3]", "[STUDENT4]", "[STUDENT5]"]
        anon_df = processor.anonymize_known_names(
            df=df, 
            text_column="text", 
            names=names, 
            replacement_names=replacement_names,
            target_text_column="anon_text"
        )

        self.assertEqual(anon_df["anon_text"][0], "Hello, my name is [TEACHER].")
        self.assertEqual(anon_df["anon_text"][1], "Hello, my name is [STUDENT2].")
        self.assertEqual(anon_df["anon_text"][2], "Hello, my name is [STUDENT3].")
        self.assertEqual(anon_df["anon_text"][3], "Hey [STUDENT5], this is [STUDENT4].")


    def test_anonymize_unknown_names(self):
        processor = TextPreprocessor()
        anon_df = processor.anonymize_unknown_names(
            df=df, 
            text_column="text", 
            target_text_column="anon_text"
        )

        regex1 = r'Hello, my name is \[PERSON\d+\].'
        self.assertRegex(anon_df["anon_text"][0], regex1)
        self.assertRegex(anon_df["anon_text"][1], regex1)

        # The following will not be caught by spacy - so be careful when using this function! If you know which names are in the dataset, use anonymize_known_names instead.
        # self.assertRegex(anon_df["anon_text"][2], regex)

        regex2 = r'Hey \[PERSON\d+\], this is \[PERSON\d+\].'
        self.assertRegex(anon_df["anon_text"][3], regex2)

    def test_merge_utterances_from_same_speaker(self):
        processor = TextPreprocessor()
        merged_df = processor.merge_utterances_from_same_speaker(
            df=speaker_df,
            text_column="text",
            speaker_column="speaker",
            target_text_column="merged_text"
        )

        self.assertEqual(merged_df["merged_text"][0], "Hey. How are you?")
        self.assertEqual(merged_df["merged_text"][1], "Good, how are you?")
        self.assertEqual(merged_df["merged_text"][2], "I'm good, thanks.")

    def test_get_speaker_text_format(self):
        processor = TextPreprocessor()

        text = processor.get_speaker_text_format(
            df=speaker_df,
            text_column="text",
            speaker_column="speaker"
        )

        target_text = """A: Hey.
A: How are you?
B: Good, how are you?
A: I'm good, thanks."""

        self.assertEqual(text, target_text)

if __name__ == "__main__":
    unittest.main()
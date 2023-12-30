import unittest
from unittest import TestCase
import pandas as pd
import math
from edu_toolkit.annotation.annotator import Annotator


class TestNeuralDiscourseAnalysis(TestCase):
    def test_get_student_reasoning(self):
        ann = Annotator()
        df = pd.DataFrame({
            "text": [
                "This is a test.", # Should be skipped over because it doesn't have enough words
                "You can instead drawing [inaudible], you can just draw six boxes.", # From the student_reasoning dataset in NCTE as reasoning
                "You can instead drawing [inaudible], you can just draw six boxes.", # Going to pretend this is from the teacher to test the speaker_value argument
                "I’m working on my words, because am I allowed to?", # From the student_reasoning dataset in NCTE as non-reasoning
            ],
            "speaker": [
                "student",
                "student",
                "teacher",
                "student"
            ]
        })

        df = ann.get_student_reasoning(
            df=df, 
            text_column="text", 
            output_column="predictions", 
            # You can also omit the speaker_column and speaker_value arguments if you want to get predictions for all rows.
            # We just include them here to show how to use them. Plus, this model should only be used on student reasoning.
            speaker_column="speaker", 
            speaker_value="student"
        )

        self.assertTrue(math.isnan(df["predictions"][0]))
        self.assertEqual(df["predictions"][1], 1)
        self.assertTrue(math.isnan(df["predictions"][2]))
        self.assertEqual(df["predictions"][3], 0)

    def test_teacher_focusing_questions(self):
        ann = Annotator()
        df = pd.DataFrame({
            "text": [
                "What do you think of when I say slope?", # 1 from the paper
                "What's the rise? What's the run?", # 0 from the paper"
            ]
        })

        df = ann.get_focusing_questions(
            df=df, 
            text_column="text", 
            output_column="predictions", 
        )
        self.assertEqual(df["predictions"][0], 1)
        self.assertEqual(df["predictions"][1], 0)

    def test_uptake(self):
        ann = Annotator()
        df = pd.DataFrame({
            "text": [
                "'Cause you took away 10 and 70 minus 10 is 60.", # High uptake from the paper
                "Why did we take away 10?",
                "test", # Utterance is too short.
                "Why did we take away 10?",
                "An obtuse angle is more than 90 degrees.", # Low uptake from the paper
                "Why don't we put our pencils down and just do some brainstorming, and then we'll go back through it?"
            ],
            "speaker": [
                "student",
                "teacher",
                "student",
                "teacher",
                "student",
                "teacher"
            ]
        })

        df = ann.get_uptake(
            df=df, 
            text_column="text", 
            output_column="predictions", 
            speaker_column="speaker",
            speaker1="student",
            speaker2="teacher",
            result_type="binary"
        )

        self.assertTrue(math.isnan(df["predictions"][0]))
        self.assertEqual(df["predictions"][1], 1)
        self.assertTrue(math.isnan(df["predictions"][2]))
        self.assertTrue(math.isnan(df["predictions"][3]))
        self.assertTrue(math.isnan(df["predictions"][4]))
        self.assertEqual(df["predictions"][5], 0)


if __name__ == "__main__":
    unittest.main()

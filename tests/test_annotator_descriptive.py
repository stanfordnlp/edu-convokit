import unittest
from unittest import TestCase
from edu_convokit import utils
import pandas as pd
import re


from edu_convokit.preprocessors.text_preprocessor import TextPreprocessor
from edu_convokit.annotation.annotator import Annotator

text_column = "text"
speaker_column = "speaker"
start_time_column = "start_timestamp"
end_time_column = "end_timestamp"

df = pd.DataFrame({
    text_column: [
        "Hello class!",
        "Hello, teacher!",
        "How is everything doing?", 
        "We're doing well.",
    ],
    speaker_column: [
        "teacher",
        "student",
        "teacher",
        "student"
    ],
    start_time_column: [
        "0",
        "5",
        "10",
        "15"
    ],
    end_time_column: [
        "5",
        "9",
        "14",
        "20"
    ]
})

total_num_words = 11
teacher_num_words = 6
student_num_words = 5

total_seconds = 18
teacher_seconds = 9
student_seconds = 9

class TestDescriptiveTextAnalysis(TestCase):
    def test_analyze_talktime(self):
        talktime_analysis_column = "talktime_analysis"
        ann = Annotator()

        talktime_df = ann.get_talktime(
            df=df,
            text_column=text_column,
            analysis_unit="words",
            output_column=talktime_analysis_column
        )

        teacher_talktime = talktime_df[talktime_df[speaker_column] == "teacher"][talktime_analysis_column].sum()
        student_talktime = talktime_df[talktime_df[speaker_column] == "student"][talktime_analysis_column].sum()
        self.assertEqual(teacher_talktime, teacher_num_words)
        self.assertEqual(student_talktime, student_num_words)

        talktime_df = ann.get_talktime(
            df=df,
            analysis_unit="timestamps",
            time_start_column=start_time_column,
            time_end_column=end_time_column,
            output_column=talktime_analysis_column
        )

        teacher_talktime = talktime_df[talktime_df[speaker_column] == "teacher"][talktime_analysis_column].sum()
        student_talktime = talktime_df[talktime_df[speaker_column] == "student"][talktime_analysis_column].sum()
        self.assertEqual(teacher_talktime, teacher_seconds)
        self.assertEqual(student_talktime, student_seconds)

if __name__ == "__main__":
    unittest.main()
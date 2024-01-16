import unittest
from unittest import TestCase
from edu_convokit import utils
import pandas as pd
import re

from edu_convokit.annotation.annotator import Annotator

class TestLexicalAnalysis(TestCase):

    def test_math_density(self):
        ann = Annotator()
        df = pd.DataFrame({
            "text": [
                "I like to eat apples.",  
                "Multiply the Multiples multiple times. And do the sums in summer. What do you think."
            ]
        })
        df = ann.get_math_density(
            df=df,
            text_column="text",
            output_column="math_density",
        )
        self.assertEqual(df['math_density'][0], 0)
        self.assertEqual(df['math_density'][1], 4) # multiply, multiples, multiple, sums

if __name__ == "__main__":
    unittest.main()
import pandas as pd
from typing import List, Union, Tuple
from collections import defaultdict
import math
from gensim.models import Phrases
from gensim.models.phrases import Phraser 
from edu_convokit import utils
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

import itertools
import re
import logging
from edu_convokit.analyzers import analyzer


class LexicalAnalyzer(analyzer.Analyzer):

    def print_word_frequency(
            self,
            text_column: str,
            topk: int = 5,
            df: pd.DataFrame = None,
            speaker_column: str = None,
            run_text_formatting: bool = False,
            run_ngrams: bool = False,
            n: int = 0
    ) -> str: 
        """
        Print word frequency for a dataframe.

        Arguments:
            df (pd.DataFrame): pandas dataframe
            text_column (str): name of column containing text to analyze
            topk (int): number of top words to return
            speaker_column (str): name of column containing speaker names. If specified, it will report word frequency for each speaker.
            run_text_formatting (bool): whether to run standard text formatting
            run_ngrams (bool): whether to run ngrams
            n (int): n for ngrams

        Returns:
            str: word frequency
        """

        text = self.report_word_frequency(
            df=df,
            text_column=text_column,
            topk=topk,
            speaker_column=speaker_column,
            run_text_formatting=run_text_formatting,
            run_ngrams=run_ngrams,
            n=n,
        )
        print(text)

    def report_word_frequency(
            self,
            text_column: str,
            topk: int = 5,
            df: pd.DataFrame = None,
            speaker_column: str = None,
            run_text_formatting: bool = False,
            run_ngrams: bool = False,
            n: int = 0
    ) -> str: 
        """
        Reports word frequency for a dataframe as a string.

        Arguments:
            df (pd.DataFrame): pandas dataframe
            text_column (str): name of column containing text to analyze
            topk (int): number of top words to return
            speaker_column (str): name of column containing speaker names. If specified, it will report word frequency for each speaker.
            run_text_formatting (bool): whether to run standard text formatting
            run_ngrams (bool): whether to run ngrams
            n (int): n for ngrams

        Returns:
            str: word frequency
        """
        if df is None: 
            df = self.get_df().copy()

        # Make sure text columns are interpreted as strings
        df[text_column] = df[text_column].astype(str)

        assert text_column in df.columns, f"Text column {text_column} not found in dataframe."

        df = df.copy()

        if run_text_formatting: # Run standard text formatting.
            df[text_column] = df[text_column].apply(utils._clean_text_to_words)

        if run_ngrams: # Run ngrams.
            df = self._compute_ngrams(df, text_column, n=n)

        df = self._format_text_column(df, text_column)

        if speaker_column is None:
            top_words = self._get_top_words(df, text_column, topk)
            text = self._format_word_frequency(top_words)
        else:
            text = "Top Words By Speaker\n"
            for speaker in df[speaker_column].unique():
                # Skip if speaker is nan
                if isinstance(speaker, float) and math.isnan(speaker):
                    continue
                speaker_df = df[df[speaker_column] == speaker]
                top_words = self._get_top_words(speaker_df, text_column, topk)
                text += f"{speaker}\n"
                text += self._format_word_frequency(top_words)
                text += "\n\n"
        return text

    def _get_top_words(
            self, 
            df: pd.DataFrame,
            text_column: str,
            topk: int = 5,
        ) -> List[Tuple[str, int]]:
        words = df[text_column].sum()
        word_counts = nltk.FreqDist(words)
        top_words = word_counts.most_common(topk)
        return top_words

    def _format_word_frequency(self, word_counts):
        text = ""
        for word, count in word_counts:
            text += f"{word}: {count}\n"
        return text

    def _format_text_column(
            self,
            df: pd.DataFrame, 
            text_column: str
        ) -> pd.DataFrame:
        """
        Format text column for lexical analysis. Check that text column is a list of strings, otherwise split on spaces.
        """
        if isinstance(df[text_column].iloc[0], str):
            df[text_column] = df[text_column].str.split()
        return df

    def _get_counts(self, texts, vocab):
        counts = {w: 0 for w in vocab}
        for split in texts:
            count = 0
            prev = ''
            for w in split:
                if w == '':
                    continue
                if w in vocab:
                    counts[w] += 1
                if count > 0: # Enable bigram counts if the vocab allows that.
                    bigram = prev + ' ' + w
                    if bigram in vocab:
                        counts[bigram] += 1
                count += 1
                prev = w
        return counts

    def _logodds(self, counts1, counts2, prior, zscore = True):
        # code from Dan Jurafsky
        # Note: counts1 will be positive and counts2 will be negative

        sigmasquared = defaultdict(float)
        sigma = defaultdict(float)
        delta = defaultdict(float)

        n1 = sum(counts1.values())
        n2 = sum(counts2.values())

        # Since we use the sum of counts from the two groups as a prior, this is equivalent to a simple log odds ratio.
        nprior = sum(prior.values())
        for word in prior.keys():
            if prior[word] == 0:
                delta[word] = 0
                continue
            l1 = float(counts1[word] + prior[word]) / (( n1 + nprior ) - (counts1[word] + prior[word]))
            l2 = float(counts2[word] + prior[word]) / (( n2 + nprior ) - (counts2[word] + prior[word]))
            sigmasquared[word] = 1/(float(counts1[word]) + float(prior[word])) + 1/(float(counts2[word]) + float(prior[word]))
            sigma[word] = math.sqrt(sigmasquared[word])
            delta[word] = (math.log(l1) - math.log(l2))
            if zscore:
                delta[word] /= sigma[word]
        return delta

    def _compute_logodds(
            self,
            df1: pd.DataFrame,
            df2: pd.DataFrame,
            text_column1: str,
            text_column2: str, 
            words2idx: dict,
        ) -> Tuple[dict, dict, dict, dict]:

        counts1 = self._get_counts(df1[text_column1], words2idx)
        counts2 = self._get_counts(df2[text_column2], words2idx)
        prior = {}
        for k, v in counts1.items():
            prior[k] = v + counts2[k]

        # Note: You might not want to z-score if there are significantly larger events in one group than the other.
        delta = self._logodds(counts1, counts2, prior, True)
        return prior, counts1, counts2, delta

    def _get_ngrams(self, text, n):
        ngrams = []
        for i in range(len(text) - n + 1):
            ngrams.append(' '.join(text[i:i+n]))
        return ngrams

    def _compute_ngrams(
            self, 
            df: pd.DataFrame,
            text_column: str, # Values in this column should be lists of strings. If they are not, they will be split on spaces.
            n: int = 0, # If n is 0, will return all ngrams.
            target_text_column: str = None, # Put ngrams in a new column.
            min_count: int = 1, # Minimum number of times an ngram must appear to be included.
    ) -> Union[List[str], pd.DataFrame]:
        df = df.copy()

        if target_text_column is None:
            target_text_column = text_column

        df[target_text_column] = df[text_column]
        df = self._format_text_column(df, target_text_column)

        if n == 0:
            ngram_model = Phrases(df[target_text_column].tolist(), min_count=min_count)
            ngram_phraser = Phraser(ngram_model)
            df[target_text_column] = df[target_text_column].apply(lambda x: ngram_phraser[x])
        else:
            df[target_text_column] = df[target_text_column].apply(lambda x: self._get_ngrams(x, n))
        return df

    def _get_logodds(
            self,
            df1: pd.DataFrame,
            df2: pd.DataFrame,
            text_column1: str,
            text_column2: str,
            topk: int = 5,
            zscore: bool = True,
            logodds_factor: float = 1.0,
            run_text_formatting: bool = False,
            run_ngrams: bool = False,
            n: int = 0,
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Return topk log-odds for each df: ([(word, log-odds), ...], [(word, log-odds), ...])

        For more information on log-odds, see: https://en.wikipedia.org/wiki/Odds_ratio

        Arguments:
            df1 (pd.DataFrame): pandas dataframe
            df2 (pd.DataFrame): pandas dataframe
            text_column1 (str): name of column containing text to analyze in df1
            text_column2 (str): name of column containing text to analyze in df2
            topk (int): number of top words to return
            zscore (bool): whether to z-score the log-odds
            logodds_factor (float): factor to multiply standard deviation by to determine top words
            run_text_formatting (bool): whether to run standard text formatting
            run_ngrams (bool): whether to run ngrams
            n (int): n for ngrams

        Returns:
            Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]: topk log-odds for each df: ([(word, log-odds), ...], [(word, log-odds), ...])
        """
        assert text_column1 in df1.columns, f"Text column {text_column1} not found in dataframe."
        assert text_column2 in df2.columns, f"Text column {text_column2} not found in dataframe."

        df1 = df1.copy()
        df2 = df2.copy()

        # Make sure text columns are interpreted as strings
        df1[text_column1] = df1[text_column1].astype(str)
        df2[text_column2] = df2[text_column2].astype(str)

        if run_text_formatting: # Run standard text formatting.
            df1[text_column1] = df1[text_column1].apply(utils._clean_text_to_words)
            df2[text_column2] = df2[text_column2].apply(utils._clean_text_to_words)

        if run_ngrams: # Run ngrams.
            df1 = self._compute_ngrams(df1, text_column1, n=n)
            df2 = self._compute_ngrams(df2, text_column2, n=n)

        df1_ = self._format_text_column(df1, text_column1)
        df2_ = self._format_text_column(df2, text_column2)

        # Get all words and build word index dictionary.
        words = df1_[text_column1].sum() + df2_[text_column2].sum()
        words = list(set(words))
        words2idx = {w: i for i, w in enumerate(words)}

        # Compute log-odds
        prior, counts1, counts2, logodds = self._compute_logodds(df1_, df2_, text_column1, text_column2, words2idx)

        LOGODDS_COLUMN = 'logodds'
        logodds_df = pd.DataFrame.from_dict(logodds, orient='index', columns=[LOGODDS_COLUMN])
        mean = 0
        std = logodds_df[LOGODDS_COLUMN].std()

        # Get top words and sort
        top_words = logodds_df[logodds_df[LOGODDS_COLUMN] >= mean + logodds_factor * std].sort_values(by=LOGODDS_COLUMN, ascending=False).head(topk)
        bottom_words = logodds_df[logodds_df[LOGODDS_COLUMN] <= mean - logodds_factor * std].sort_values(by=LOGODDS_COLUMN, ascending=True).head(topk)

        return list(zip(top_words.index, top_words[LOGODDS_COLUMN])), list(zip(bottom_words.index, bottom_words[LOGODDS_COLUMN]))

    def _format_log_odds(self, log_odds1, log_odds2):
        text = ""
        text += "Top words for Group 1\n"
        for word, score in log_odds1:
            text += f"{word}: {score}\n"
        text += "\n\n"
        text += "Top words for Group 2\n"
        for word, score in log_odds2:
            text += f"{word}: {score}\n"
        return text
    
    def report_log_odds(
            self,
            df1: pd.DataFrame,
            df2: pd.DataFrame,
            text_column1: str,
            text_column2: str,
            topk: int = 5,
            zscore: bool = True,
            logodds_factor: float = 1.0,
            run_text_formatting: bool = False,
            run_ngrams: bool = False,
            n: int = 0,
    ) -> str:
        """
        Return formatted topk log-odds for each df: ([(word, log-odds), ...], [(word, log-odds), ...])

        Arguments:
            df1 (pd.DataFrame): pandas dataframe
            df2 (pd.DataFrame): pandas dataframe
            text_column1 (str): name of column containing text to analyze in df1
            text_column2 (str): name of column containing text to analyze in df2
            topk (int): number of top words to return
            zscore (bool): whether to z-score the log-odds
            logodds_factor (float): factor to multiply standard deviation by to determine top words
            run_text_formatting (bool): whether to run standard text formatting
            run_ngrams (bool): whether to run ngrams
            n (int): n for ngrams

        Returns:
            str: formatted topk log-odds for each df: ([(word, log-odds), ...], [(word, log-odds), ...])
        """

        log_odds1, log_odds2 = self._get_logodds(
            df1=df1,
            df2=df2,
            text_column1=text_column1,
            text_column2=text_column2,
            topk=topk,
            zscore=zscore,
            logodds_factor=logodds_factor,
            run_text_formatting=run_text_formatting,
            run_ngrams=run_ngrams,
            n=n,
        )

        text = self._format_log_odds(log_odds1, log_odds2)
        return text
    
    def print_log_odds(
            self,
            df1: pd.DataFrame,
            df2: pd.DataFrame,
            text_column1: str, 
            text_column2: str,
            topk: int = 5,
            zscore: bool = True,
            logodds_factor: float = 1.0,
            run_text_formatting: bool = False,
            run_ngrams: bool = False,
            n: int = 0,
    ) -> None:
        """
        Print topk log-odds for each df: ([(word, log-odds), ...], [(word, log-odds), ...])

        Arguments:
            df1 (pd.DataFrame): pandas dataframe
            df2 (pd.DataFrame): pandas dataframe
            text_column1 (str): name of column containing text to analyze in df1
            text_column2 (str): name of column containing text to analyze in df2
            topk (int): number of top words to return
            zscore (bool): whether to z-score the log-odds
            logodds_factor (float): factor to multiply standard deviation by to determine top words
            run_text_formatting (bool): whether to run standard text formatting
            run_ngrams (bool): whether to run ngrams
            n (int): n for ngrams
        """

        text = self.report_log_odds(
            df1=df1,
            df2=df2,
            text_column1=text_column1,
            text_column2=text_column2,
            topk=topk,
            zscore=zscore,
            logodds_factor=logodds_factor,
            run_text_formatting=run_text_formatting,
            run_ngrams=run_ngrams,
            n=n,
        )

        print(text)

    def plot_log_odds(
            self,
            df1: pd.DataFrame,
            df2: pd.DataFrame,
            text_column1: str,
            text_column2: str,
            group1_name: str = "Group 1",
            group2_name: str = "Group 2",
            topk: int = 5,
            save_path: str = None,
            zscore: bool = True,
            logodds_factor: float = 1.0,
            run_text_formatting: bool = False,
            run_ngrams: bool = False,
            n: int = 0,
    ) -> None:
        """
        Plot topk log-odds for each df: ([(word, log-odds), ...], [(word, log-odds), ...])

        Arguments:
            df1 (pd.DataFrame): pandas dataframe
            df2 (pd.DataFrame): pandas dataframe
            text_column1 (str): name of column containing text to analyze in df1
            text_column2 (str): name of column containing text to analyze in df2
            group1_name (str): name of group 1
            group2_name (str): name of group 2
            topk (int): number of top words to return
            save_path (str): path to save plot
            zscore (bool): whether to z-score the log-odds
            logodds_factor (float): factor to multiply standard deviation by to determine top words
            run_text_formatting (bool): whether to run standard text formatting
            run_ngrams (bool): whether to run ngrams
            n (int): n for ngrams
        """

        sns.set_theme(style="whitegrid")
        sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
        plt.rcParams["font.family"] = "serif"

        log_odds1, log_odds2 = self._get_logodds(
            df1=df1,
            df2=df2,
            text_column1=text_column1,
            text_column2=text_column2,
            topk=topk,
            zscore=zscore,
            logodds_factor=logodds_factor,
            run_text_formatting=run_text_formatting,
            run_ngrams=run_ngrams,
            n=n,
        )

        # Create dataframe
        log_odds_df = pd.DataFrame(log_odds1 + log_odds2, columns=['word', 'log_odds'])
        # Plot  x-axis: log-odds, y-axis: words
        plt.figure(figsize=(6, len(log_odds_df) / 2))
        sns.barplot(x='log_odds', y='word', data=log_odds_df)
        plt.xlabel('Log odds')
        plt.ylabel('Words')

        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        plt.text(x_min, y_min, group2_name, ha='left', va='center') # second group because it's negative
        plt.text(x_max, y_min, group1_name, ha='right', va='center')
        plt.title(f"Log odds: {group1_name} vs. {group2_name}")
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.clf()

import os
import pandas as pd
import json
import re
import nltk
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
from edu_convokit.constants import VALID_FILE_EXTENSIONS
import numpy as np
import pkg_resources


def load_text_file(filepath):
    """
    Loads a text file and returns the text as a string.
    """
    try: 
        content = pkg_resources.resource_string(__name__, filepath)
        return content.decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filepath} not found.")

def split_dataframe(df, num_bins): # output list of dfs
    """
    Splits a dataframe into num_bins chunks
    """
    assert isinstance(df, pd.DataFrame), "df must be a pandas dataframe"
    assert isinstance(num_bins, int) and num_bins > 0, "num_bins must be a positive integer"
    return np.array_split(df, num_bins)


def is_valid_analysis_file_extension(fname):
    """
    Checks if a filename has a valid extension for analysis (csv)
    """
    return fname.endswith(".csv")

def get_random_file(dirpath):
    # Get all files in dir path that are valid analysis files
    files = [f for f in os.listdir(dirpath) if is_valid_analysis_file_extension(f)]
    # Return random file
    return files[np.random.randint(len(files))]

def is_valid_file_extension(fname):
    """
    Checks if a filename has a valid extension.
    """
    return any([fname.endswith(ext) for ext in VALID_FILE_EXTENSIONS])

def convert_time_hh_mm_ss_to_seconds(time):
    """
    Converts a time in the format HH:MM:SS to seconds
    """
    assert isinstance(time, str) and len(time.split(":")) == 3, "Time must be in the format HH:MM:SS"
    hours, minutes, seconds = time.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)

def xlsx_to_csv(xlsx_fname, replace_xlsx=False):
    """
    Converts an xlsx file to csv.
    """
    csv_fname = xlsx_fname.replace(".xlsx", ".csv")
    df = pd.read_excel(xlsx_fname)
    df.to_csv(csv_fname, index=False)
    if replace_xlsx:
        os.remove(xlsx_fname)
    return csv_fname

def _clean_text_to_words(
        text: str,
        remove_stopwords: bool = True,
        remove_numeric: bool = True,
        stem: bool = False,
        remove_short: bool = True,
    ):
    sno = nltk.stem.SnowballStemmer('english')
    punct_chars = list((set(string.punctuation) | {'’', '‘', '–', '—', '~', '|', '“', '”', '…', "'", "`", '_'}) - set(['#']))
    punct_chars.sort()
    punctuation = ''.join(punct_chars)
    printable = set(string.printable)

    # lower case
    text = text.lower()
    # eliminate urls
    text = re.sub(r'http\S*|\S*\.com\S*|\S*www\S*', ' ', text)
    # substitute all other punctuation with whitespace
    replace = re.compile('[%s]' % re.escape(punctuation))
    text = replace.sub(' ', text)
    # replace all whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    # strip off spaces on either end
    text = text.strip()
    # make sure all chars are printable
    text = ''.join([c for c in text if c in printable])
    words = text.split()
    if remove_stopwords:
        words = [w for w in words if w not in stopwords.words('english')]
    if remove_numeric:
        words = [w for w in words if not w.isdigit()]
    if stem:
        words = [sno.stem(w) for w in words]
    if remove_short:
        words = [w for w in words if len(w) >= 3]
    return words

def load_data(fname): 
    # if fname.endswith(".xlsx") -> load with pandas
    if fname.endswith(".xlsx"):
        return pd.read_excel(fname)
    # if fname.endswith(".csv") -> load with pandas
    elif fname.endswith(".csv"):
        return pd.read_csv(fname)
    # if fname.endswith(".json") -> load then cast as dataframe
    elif fname.endswith(".json"):
        with open(fname) as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"File type {fname.split('.')[-1]} not supported. Feel free to add support for this file type!")

def get_valid_analysis_files_in_dir(dirname):
    return [os.path.join(dirname, _) for _ in os.listdir(dirname) if is_valid_analysis_file_extension(_)]

def merge_dataframes_in_list(filenames, max_transcripts=None):
    df = pd.DataFrame()
    for filename in filenames:
        df = pd.concat([df, load_data(filename)], axis=0)
        if max_transcripts is not None and len(df) >= max_transcripts:
            break
    return df

def merge_dataframes_in_dir(dirname, max_transcripts=None):
    filenames = get_valid_analysis_files_in_dir(dirname)
    df = merge_dataframes_in_list(filenames, max_transcripts=max_transcripts)
    return df
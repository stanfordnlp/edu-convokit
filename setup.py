import os
import re

import setuptools

here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, "edu_toolkit", "__init__.py")) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

extras_require = {
    "latex": ("bibtexparser",),
}

setuptools.setup(
    name="edu-toolkit",
    packages=setuptools.find_packages(exclude=["tests"]),
    version=version,
    license="MIT",
    description="edu-toolkit: An Open-Source Framework for Education Language Data",
    author="Rose E. Wang",
    author_email="rewang@cs.stanford.edu",
    url="https://github.com/rosewang2008/edu-toolkit",
    install_requires=[
        "tqdm",
        "numpy",
        "scipy",
        "nltk",
        "torch",
        "transformers",
        "scipy",
        "clean-text",
        "openpyxl",
        "spacy",
        "gensim",
        "num2words==0.5.10",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pandas>=2.0.0",
        "clean-text",
    ],
    extras_require=extras_require,
    python_requires="~=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
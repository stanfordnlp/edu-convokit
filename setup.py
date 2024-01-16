import os
import re

import setuptools

here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, "edu_convokit", "__init__.py")) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

extras_require = {
    "latex": ("bibtexparser",),
    "tests": ("pytest",),
}

with open('README.md') as f:
    readme = f.read()

setuptools.setup(
    name="edu_convokit",
    packages=setuptools.find_packages(exclude=["tests"]),
    version=version,
    license="MIT",
    description="Edu-ConvoKit: An Open-Source Framework for Education Conversation Data",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Rose E. Wang",
    author_email="rewang@cs.stanford.edu",
    url="https://github.com/rosewang2008/edu-convokit",
    project_urls={
        "Homepage": "https://rosewang2008.github.io/edu-convokit/",
        "Documentation": "https://edu-convokit.readthedocs.io/en/latest/",
        "Source": "https://github.com/rosewang2008/edu-convokit/",
    },
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
        "pandas",
        "clean-text",
        "tiktoken"
    ],
    extras_require=extras_require,
    tests_require=["pytest"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Typing :: Typed",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Utilities",
    ],
    package_data={"edu_convokit": ["prompts/conversation/*.txt", "prompts/utterance/*.txt"]},
    include_package_data=True,
    python_requires=">=3.10,<3.12",
)

<p align="center">
  <img src="assets/full_logo.png" width="600"/>
</p>

<h1><img src="assets/logo.png" height="30" /> Edu-ConvoKit: An Open-Source Framework for Education Conversation Data </h1>

The **Edu-ConvoKit** is an open-source framework designed to facilitate the study of conversation language data in educational settings.
It provides a practical and efficient pipeline for essential tasks such as text pre-processing, annotation, and analysis, tailored to meet the needs of researchers and developers.
This toolkit aims to enhance the accessibility and reproducibility of educational language data analysis, as well as advance both natural language processing (NLP) and education research.
By simplifying these key operations, the Edu-ConvoKit supports the efficient exploration and interpretation of text data in education.

## 📖 Table of Contents
[**Installation**](#installation) | [**Tutorials**](#tutorials) | [**Example Usage**](#example-usage) | [**Documentation**](https://edu-convokit.readthedocs.io/en/latest/) | [**Citation**](#citation) | [**Contact**](#contact)

## Installation

You can directly install the **edu-convokit** from GitHub:

```bash

pip install edu-convokit

```

## Overview of the `edu-convokit` Pipeline

<p align="center">
  <img src="assets/overview.png" width="600"/>
</p>


The **Edu-ConvoKit** pipeline consists of three key modules: `preprocess`, `annotate`, and `analyze`.
The pipeline is designed to be modular, so you can use any combination of these modules to suit your needs.
This pipeline has also been used by prior work in the field of education, so you can easily reproduce their results. 

<p align="center">
  <img src="assets/main_figure.png"/>
</p>

## Tutorials

We have provided a series of tutorials to help you get started with the **edu-convokit**.

### Basics of `edu-convokit`

There are three key modules of the **edu-convokit** pipeline: `preprocess`, `annotate`, and `analyze`.

* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][textcolab] [Tutorial: Text Pre-processing][textcolab]
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][annotationcolab] [Tutorial: Annotation][annotationcolab]
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][analyzecolab] [Tutorial: Analysis][analyzecolab]
...

### Datasets with `edu-convokit`

We've applied the **edu-convokit** to a variety of datasets. Here are some examples:
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][nctecolab] [Tutorial: NCTE Dataset][nctecolab]
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][ambercolab] [Tutorial: Amber Dataset][ambercolab]
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][talkmovescolab] [Tutorial: Talk Moves Dataset][talkmovescolab]

## Papers that have used the **edu-convokit**

Please find [here](papers.md) a list of papers that have used the **edu-convokit**.

## Citation

If you use the **edu-convokit** in your research, please cite the following paper:

```
Coming soon...
```

If you would like to be added to the list of papers that have used the **edu-convokit**, please make a pull request or contact Rose E. Wang at rewang@cs.stanford.edu.

[textcolab]: https://colab.research.google.com/drive/1a-EwYwkNYHSNcNThNTXe6DNpsis0bpQK
[annotationcolab]: https://colab.research.google.com/drive/1rBwEctFtmQowZHxralH2OGT5uV0zRIQw 
[analyzecolab]: https://colab.research.google.com/drive/1xfrq5Ka3FZH7t9l87u4sa_oMlmMvuTfe 
[ambercolab]: https://colab.research.google.com/drive/1Q3anUPcemMils4cz2gwEwDdKCjEdm6T9 
[talkmovescolab]: https://colab.research.google.com/drive/1qt_S3GjxIwXk6ONztbYAHeX8WHy1uxDd 
[nctecolab]: https://colab.research.google.com/drive/1k3fn6uY4QRMtPUZN6hpMd6o-0g7fYotg 


## Contact

If you have any questions, please contact Rose E. Wang at rewang@cs.stanford.edu.
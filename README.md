# Implementation of AdPXT framework

This is the anonymized github repository for the submission, entitled **"Deep Adversarial Prefix Tuning for Domain Generalization in Text Classification"**.
For reproducibility, the codes and datasets are publicly available during the review phase.

## Run the codes

- hugging face transformers
- python
- pytorch
- numpy


## Unzip data.zip
- replace sentiment analysis dataset to data/TextClassification/
- replace Natural Language Inference dataset to data/NLI/

```
pip install -r requirement.txt
```
You can simply run the code with the default setting by the following command:

```
python train.py -c config_nli_adpxt.json
```

## Datasets
- Sentiment Analysis
  - Amazon Review
  - IMDB
  - SST-2
- Natural Language Inference
  - MNLI
  - SNLI
  - SICK

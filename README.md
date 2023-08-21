# Implementation of AdPXT framework

This is the anonymized github repository for the submission, entitled **"Deep Adversarial Prefix Tuning for Domain
Generalization in Text Classification"**.
For reproducibility, the codes and datasets are publicly available during the review phase.

## Run the codes

- hugging face transformers
- python
- pytorch
- numpy

## Unzip data.zip

## Requirements

```
pip install -r requirement.txt
```

## Usage

```
You can simply run the code with the default setting by the following command:
python train.py -c config_nli_adpxt.json
```

## Datasets

- Sentiment Analysis
    - Amazon Review(origin): https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
        - subset of Amazon review dataset: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
            - John Blitzer, Mark Dredze, Fernando Pereira. Biographies, Bollywood, Boom-boxes and Blenders: Domain
              Adaptation for Sentiment Classification. Association of Computational Linguistics (ACL), 2007.
    - IMDB: https://ai.stanford.edu/~amaas/data/sentiment/
        - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning
          Word Vectors for Sentiment Analysis. In Proceedings of the 49th Annual Meeting of the Association for
          Computational Linguistics: Human Language Technologies, pages 142–150, Portland, Oregon, USA. Association for
          Computational Linguistics.
    - SST-2: https://nlp.stanford.edu/sentiment/
        - Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher
          Potts. 2013. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. In Proceedings of
          the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642, Seattle, Washington,
          USA. Association for Computational Linguistics.

- Natural Language Inference
    - MNLI: https://cims.nyu.edu/~sbowman/multinli/
        - Adina Williams, Nikita Nangia, and Samuel Bowman. 2018. A Broad-Coverage Challenge Corpus for Sentence
          Understanding through Inference. In Proceedings of the 2018 Conference of the North American Chapter of the
          Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages
          1112–1122, New Orleans, Louisiana. Association for Computational Linguistics.
    - SNLI: https://nlp.stanford.edu/projects/snli/
        - Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015.
          A large annotated corpus for learning natural language inference.
          Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP).
    - SICK: https://marcobaroni.org/composes/sick.html
        - Marco Marelli, Stefano Menini, Marco Baroni, Luisa Bentivogli, Raffaella Bernardi, and Roberto Zamparelli.
          2014. A SICK cure for the evaluation of compositional distributional semantic models. In Proceedings of the
          Ninth International Conference on Language Resources and Evaluation (LREC'14), pages 216–223, Reykjavik,
          Iceland. European Language Resources Association (ELRA).

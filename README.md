# Autoregressive Score Generation for Multi-trait Essay Scoring (ArTS)
This repository is the implementation of the ArTS architecture, introduced in the paper, [Autoregressive Score Generation for Multi-trait Essay Scoring (EACL Findings 2024)](https://aclanthology.org/2024.findings-eacl.115/).


## Citation
Please cite our paper if you find this repository helpful 😊
```
@inproceedings{do-etal-2024-autoregressive,
    title = "Autoregressive Score Generation for Multi-trait Essay Scoring",
    author = "Do, Heejin  and
      Kim, Yunsu  and
      Lee, Gary",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-eacl.115/",
    pages = "1659--1666",
    abstract = "Recently, encoder-only pre-trained models such as BERT have been successfully applied in automated essay scoring (AES) to predict a single overall score. However, studies have yet to explore these models in multi-trait AES, possibly due to the inefficiency of replicating BERT-based models for each trait. Breaking away from the existing sole use of *encoder*, we propose an autoregressive prediction of multi-trait scores (ArTS), incorporating a *decoding* process by leveraging the pre-trained T5. Unlike prior regression or classification methods, we redefine AES as a score-generation task, allowing a single model to predict multiple scores. During decoding, the subsequent trait prediction can benefit by conditioning on the preceding trait scores. Experimental results proved the efficacy of ArTS, showing over 5{\%} average improvements in both prompts and traits."
}
```

## Package Requirements
We conducted experiments in the environment below, but the same version may not be required.

- python '3.8.0'
- pytorch '2.0.1+cu118'
- numpy '1.23.5'
- wandb '0.15.12'
- pandas '2.0.3'
- transformers '4.31.0'
- datasets '2.11.0'
- scikit-learn '1.2.2'


## Training
- bash runs/train_trainer_arts.sh

## Evaluation
- bash runs/eval_arts.sh


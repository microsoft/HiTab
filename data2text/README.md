# Data-to-Text Generation 

We explore four baseline models to generate meaning text from hierarchical tables in HiTab. 
Three of them are transformer-based models: T5, BART, and BERT-to-BERT. The other is a Pointer-Generator Network based on LSTM architecture. 

## [0] Preliminaries 
To start with, make sure to install the following requirements: 
```
pip install openpyxl
pip install datasets 
pip install transformers 
```


## [1] Data Pre-processing 
Read in the `train_samples.jsonl`, `dev_samples.jsonl`, `test_samples.jsonl` in the `./data/` directory. 

Process each sample with: (1) highlighted/linked table cells, (2) with additional operations and answer(s). 
- The generation `target` label is the annotated `sub_sentence`. 
- To create a serialized table data input, we need to: (1) find all linked entity/quantity cells, (2) find all of their ascendants, then linearize their cell contents following a top-down left-to-right order. If extra operational information is required, we will then append the answer formula and answer string to the `source` as the final model input. 

This process create pairs of source-target for train/dev/test sets. 
To perform data pre-processing for the **cell highlight** setting, simply run: 
```bash
python do_preprocess.py
```
Or to enable the **cell & calculation** setting, specify the additional argument by: 
```bash
python do_preprocess.py --add_aggr 
```
Both will load the data from `hitab/data/` directory and generate a processed version in `hitab/data2text/data/`.

Note that the input samples require a another layer of tokenization, using `hitab/data2text/experiment/pointer_generator/parse_sample.py`. 


## [2] Experiment: Training and Evaluation 

The `experiment` directory contains the code for training (`train_d2t.py`) and evaluation (`eval_d2t.py`).
The T5, BART, and BERT-to-BERT directly call the training process from the installed [`transformers`](https://github.com/huggingface/transformers) library. 
Pointer-Generator Network (PGN) requires additional code modules, specifically in the `pointer_generator` directory. 

To follow the training pipeline, take BART for an example, run: 
```bash
python run_experiment.py --expr_name bart --do_train --do_eval --do_test 
```
Alter the `expr_name` argument among t5/bart/b2b/pgn to explore different models. 
# HiTab : A Hierarchical Table Dataset for Question Answering and Natural Language Generation

HiTab is a dataset for question answering and data-to-text over hierarchical tables . It contains 10,672 samples and 3,597 tables from statistical reports ([StatCan](https://www.statcan.gc.ca/), [NSF](https://www.nsf.gov/)) and Wikipedia ([ToTTo](https://github.com/google-research-datasets/ToTTo)).  98.1% of the tables in HiTab are with hierarchies.  You can find more details in [our paper](https://arxiv.org/abs/2108.06712).

During the dataset annotation process, annotators first manually collect tables and  descriptive sentences highly-related to tables on statistical websites written by professional analysts. And then these descriptions are revised to questions to preserve the original meanings and analyses.

We hope HiTab can serve as a useful benchmark for table understanding on hierarchical tables. 


### Note
In the latest version dataset, we have improved the algorithm for hierarchy extraction and fixed some unreliable question answering pairs, thus the qa and data2text performance will be slightly higher than the results reported in the paper. We show more details in qa and data2text descriptions.


## :beers: Updates

+ **Stay tuned**: Code of data2text.
+ **2021-2-24**: HiTab is accepted by ACL 2022 main conference.
+ **2021-2-7**: We released the final version of HiTab data. Please feel free to explore it!
+ **2021-12-6**: We released code of question answering and a new version HiTab data. 
Several modifications on data: (1) more precise hierarchies are derived for \~3\% tables with new heuristic algorithms; 
(2) fix the problem that \~0.6\% tables ranges were not correctly extracted from original excel file; 
(3) temporarily set aside \~1.5\% samples for further check containing unreliable answers or aggregations, which hopefully won't affect evaluating new methods due to the small proportion. 
We'll release the final version HiTab version after checking. Thank you for your patience.
+ **2021-9-2**: We released full HiTab data, including (1) question answering and data2text samples, (2) tables with parsed hierarchies.


## Dataset Description

HiTab dataset consists of three `.jsonl` files for train/dev/test samples and a directory of `.json` files for tables.

### Sample Format

```json
{
  "id": "7392822961051524760",
  "table_id": "1028",
  "table_source": "statcan",
  "sentence_id": "5895",
  "sub_sentence_id": "1",
  "sub_sentence": "in 2013/2014, on any given day, there were on average 139,337 adult offenders being supervised in either provincial/territorial or federal correctional services",
  "question": "in 2013/2014, on any given day, how many adult offenders are being supervised in either provincial/territorial or federal correctional services?",
  "answer": [
    139337
  ],
  "aggregation": [
    "sum"
  ],
  "linked_cells": {
    "entity_link": {
      "top": {
        "correctional services": {
          "(0, 7)": "total correctional services"
        }
      },
      "left": {
        "provincial/territorial": {
          "(14, 0)": "provinces and territories - total"
        },
        "federal": {
          "(15, 0)": "federal"
        }
      },
      "top_left_corner": {}
    },
    "quantity_link": {
      "[ANSWER]": {
        "(15, 7)": 22895.0,
        "(14, 7)": 116442.0
      }
    }
  },
  "answer_formulas": [
    "=H17+H18"
  ],
  "reference_cells_map": {
    "H17": "(14, 7)",
    "H18": "(15, 7)"
  }
}
```

+ **Meta Data**: `id` is the unique id of each sample. The other ids describe the detailed information in annotations and `table_source` shows which source the table comes from. 
+ **Task Data**:  `sub_sentence` is "text" in data2text task. `question` and `answer` are for question answering task.
+ **Links and Compositions**: `aggregation` is the aggregation(s) to derive the answer. `linked_cells` are the regarded cells in both tasks. `answer_formulas` are formulas about how cells composite to derive the answer. `reference_cells_map` are the referenced cells to current cell coordinate in the table matrix. 
  + **Linked Cells**: `linked_cells` are divided into `entity_link` (not in data region) and `quantity_link` (cells in data region). `entity_link` are further classified into `top` (top header), `left` (left header) and `top-left-corner` (on the top-left corner of table). The **key** of each link is the phrase in the sub-sentence, like *"correctional services"*. The **value** contains key-value pairs in format **cell coordinate - cell string** in table, like *"(0, 7)": "total correctional services"* .  *[ANSWER]* is a special key as it stands for the cells that composite to derive the answer. Usually *[ANSWER]* appears in `quantity_link`, but sometimes it can be in `entity_link` if the answer is a header.

The cell coordinates above are under the coordinate system of the table matrix provided in following table format.


### Table Format

```json
{
  "top_root": {
    "row_index": -1,
    "column_index": -1,
    "children": [
      {
        "row_index": 0,
        "column_index": 1,
        "children": [
          {
            "row_index": 1,
            "column_index": 1,
            "children": []
          },
          {
            "row_index": 1,
            "column_index": 2,
            "children": []
          }
        ]
      },...
    ]
  },
  "left_root": {
    "row_index": -1,
    "column_index": -1,
    "children": [
      {
        "row_index": 2,
        "column_index": 0,
        "children": [
          {
            "row_index": 3,
            "column_index": 0,
            "children": []
          },
          {
            "row_index": 4,
            "column_index": 0,
            "children": []
          },...
        ]
      },
      ...
    ]
  },
  "top_header_rows_num": 3,
  "left_header_columns_num": 1
}
```

`top_root` and `left_root` are the parsed tree hierarchies of top headers and left headers. `row_index` and `column_index` are row and column index of current header node in the table matrix. *-1* stands for the virtual root. `top_header_rows_num` and `left_header_columns_num` are number of rows/columns of headers in the table matrix.





```json
{ 
  "texts": [
    [
      "",
      "total beverages",
      "",
      "skim, 1% or 2% milk",
      "",
      "whole milk and flavoured milk",
      "",
      "fruit juice",
      "",
      "soft drinks",
      "",
      "fruit drinks",
      ""
    ],...
  ],
  "merged_regions": [
    {
      "first_row": 0,
      "last_row": 0,
      "first_column": 5,
      "last_column": 6
    },
    {
      "first_row": 0,
      "last_row": 0,
      "first_column": 3,
      "last_column": 4
    }, ...
  ],
}
```

`texts` is the complete table matrix consisting M rows and N columns. `merged_regions` lists all the merged cells. If a cell is a merged cells, only its **core cell**  (the top left position in the merged cell) will have content in `texts`, and others will be empty.

The tables in `tables/hmt/` directory are an adapted version to the hierarchical matrix table data structure customized for hierarchy-aware logical form, which basically contain the same information as the data format above.


## Question Answering

The question answering codebase references [pytorch version of MAPO](https://github.com/pcyin/pytorch_neural_symbolic_machines) 
and [TaBERT](https://github.com/facebookresearch/TaBERT). Many respects and thanks for [PengCheng Yin](https://pcyin.me/)'s great work!

Weakly supervised Table QA usually requires consistent programs for warm start and alignments between question and table schemas or headers as input features,
which we already provide as `data/explore/saved_programs.json`, and `data/processed_input/`. 

Users can also start with raw data format, i.e. `data/*_samples.jsonl`, by searching programs with `qa/table/random_explore.py` and extract question-table alignments with `qa/datadump/process_input.py`. The detailed usage of console arguments can be found in the code files.


### Quick Start
Here is a very quick start script for "MAPO with hierarchical-aware logical form" method in HiTab paper using our processed data.
```shell
# unzip table files
unzip -d data/ data/tables.zip
# set 'MY_PATH_TO' in config as the path to the project (similarly for partial supervision)
vim qa/config/config.vanilla_bert.json
# train
bash train_hmtqa.sh
# test
bash test_hmtqa.sh
```
The training phase takes \~10 hours on 4 V100 GPUs. 

If needed, we provide the baseline "MAPO with hierarchical-aware logical form" [model checkpoint](https://drive.google.com/file/d/1_S5yQ2gKH7U3v-7Aa7m55NL1lmxmhh4U/view?usp=sharing), which achieves 45.5% on dev set and 42.3% on test set. Both are sligtly higher than the results in paper due to the updated dataset. We also find that disabling trigger words in training may increase accuracy at the cost of much higher spurious program rate, thus we choose to retain the trigger words.


## Data2text

[TODO]


## Reference

If you find HiTab dataset is useful in your work, please consider citing the paper:

```
@article{cheng2021hitab,
  title={HiTab: A Hierarchical Table Dataset for Question Answering and Natural Language Generation},
  author={Cheng, Zhoujun and Dong, Haoyu and Wang, Zhiruo and Jia, Ran and Guo, Jiaqi and Gao, Yan and Han, Shi and Lou, Jian-Guang and Zhang, Dongmei},
  journal={arXiv preprint arXiv:2108.06712},
  year={2021}
}
```



## License

This dataset follows the Computational Use of Data Agreement v1.0.



## Contact

If you have any question regarding HiTab dataset or publication, please create an issue in this repository.  You can also reach us by e-mail addresses in the paper.


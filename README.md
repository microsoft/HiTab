# HiTab : A Hierarchical Table Dataset for Question Answering and Natural Language Generation

HiTab is a dataset for question answering and data-to-text over hierarchical tables . It contains 10,674 samples and 3,597 tables from statistical reports ([StatCan](https://www.statcan.gc.ca/), [NSF](https://www.nsf.gov/)) and Wikipedia ([ToTTo](https://github.com/google-research-datasets/ToTTo)).  98.1% of the tables in HiTab are with hierarchies.  You can find more details in [our paper](https://arxiv.org/abs/2108.06712).

During the dataset annotation process, annotators first manually collect tables and  descriptive sentences highly-related to tables on statistical websites written by professional analysts. And then these descriptions are revised to questions to preserve the original meanings and analyses.

We hope HiTab can serve as a useful benchmark for table understanding on hierarchical tables. 



## :beers: Updates

+ **Stay tuned!**: Code of question answering and data2text.

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
  "key_part": "139,337",
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
+ **Task Data**:  `sub_sentence` is "text" in data2text task. `key_part` is the questioned part in sub-sentence. `question` and `answer` are for question answering task.
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





```
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

`texts` is the complete table matrix consisting $M$ rows and $N$ columns. `merged_regions` lists all the merged cells. If a cell is a merged cells, only its **core cell**  (the top left position in the merged cell) will have content in `texts`, and others will be empty.



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


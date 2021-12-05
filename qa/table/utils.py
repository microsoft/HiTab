import re
import math
import numpy as np
import babel
from babel import numbers

from qa.datadump.utils import naive_str_to_float


def hmt_score(prediction, answer):
    prediction = hmt_process_answer(prediction)
    answer = hmt_process_answer(answer)
    if hmt_equal(prediction, answer):
        return 1.0
    else:
        return 0.0


def hmt_process_answer(answer):
    """ 4 types of answer: 1)region; 2)num_list(aggr); 3)header_list(argmax); 4)num(count/div)"""
    if isinstance(answer, int) or isinstance(answer, float):
        return float(answer)
    if isinstance(answer, str):
        return naive_str_to_float(answer.strip().lower())
    if isinstance(answer, list):
        if isinstance(answer[0], list):  # pred region
            if len(answer) == 1 and len(answer[0]) == 1:  # pred region with one cell, flatten
                return hmt_process_answer(answer[0][0])
            elif len(answer) == 1:  # pred region with one line
                return hmt_process_answer(answer[0])
            elif len(answer[0]) == 1:  # pred region with one line
                return hmt_process_answer([row[0] for row in answer])
            else:  # pred region is a matrix
                return [hmt_process_answer(a) for a in answer]
        else:  # list or processed single-line region
            if len(answer) == 1:  # answer with one cell or pred list
                return hmt_process_answer(answer[0])
            else:
                return [hmt_process_answer(a) for a in answer]


def hmt_equal(prediction, answer):
    if type(prediction) != type(answer):
        return False
    if isinstance(prediction, str):
        return prediction == answer
    if isinstance(prediction, int) or isinstance(prediction, float):
        return math.fabs(prediction - answer) < 1e-5
    if isinstance(prediction, list):
        if len(prediction) != len(answer):
            return False
        return all([hmt_equal(prediction[i], answer[i]) for i in range(len(prediction))])
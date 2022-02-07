AGGR_MAP = {
    'sum': ['sum'],
    'average': ['average'],
    'div': ['proportion', 'difference_rate'],
    'diff': ['difference', 'difference_rate'],
    'max': ['max', 'argmax'],
    'min': ['min', 'argmin'],
    'argmax': ['argmax'],
    'argmin': ['argmin'],
    'pair-argmax': ['argmax'],
    'pair-argmin': ['argmin'],
    'counta': ['count'],
    'opposite': ['opposite'],
    'topk-argmax': ['argmax'],  # TODO: following may not be precisely matched
    'topk-argmin': ['argmin'],
    'kth-argmax': ['argmax'],
    'kth-argmin': ['argmin'],
    'range': ['max', 'min', 'argmax', 'argmin'],
    'greater_than': ['argmax'],
    'less_than': ['argmin']
}

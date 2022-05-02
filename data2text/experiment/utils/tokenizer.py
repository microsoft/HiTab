"""Tokenizer utilities for experiments. 

Possible tokenizer names are: 
 - t5-base, t5-large
 - bart-base, bart-large
 - bert-base-uncased, bert-large-uncased 
"""

from transformers import AutoTokenizer

import logging 
logger = logging.getLogger(__name__) 

special_tokens_dict = {
    'sep_token': '<sep>', 
    'cls_token': '<cls>', 
    'mask_token': '<mask>'
}

new_tokens = ['<title>', '<cell>', '<agg>', '<top>', '<left>', '<corner>', '<data>']


def prepare_tokenizer(name: str, verbose: bool = False) -> AutoTokenizer: 
    """Prepare the loaded tokenizer class given the (model) name. 
    args: 
        name: <str>, key of the specified tokenizer 
              choices = ['t5-base', 'bart-base', 'bert-base-uncased'] 
        verbose: <bool>, whether in a verbose mode 
    rets: 
        tokenzier: an automatically identified tokenizer class. 
    """

    tokenizer = AutoTokenizer.from_pretrained(name)
    # tokenizer.add_special_tokens(special_tokens_dict)
    # tokenizer.add_tokens(new_tokens)

    if verbose == True:
        logger.info(f'[utils >> prepare_tokenizer] gets tokenizer from name [{name}]')
        # logger.info(f'[utils >> prepare_tokenizer] adds special tokens {list(special_tokens_dict.keys())}')

    return tokenizer

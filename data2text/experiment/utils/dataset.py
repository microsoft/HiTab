"""Load train/dev and test sets for experiments."""

from typing import Dict 
from datasets import load_dataset


special_tokens_map = {
    't5': {'cls': 1, 'sep': 1},       # eos_token_id
    'bart': {'cls': 0, 'sep': 2},     # cls_token_id, sep_token_id
    'b2b': {'cls': 101, 'sep': 102},  # cls_token_id, sep_token_id
}


# %% train

def get_sample_naive(sample, tokenizer, args):
    """Tokenize a sample to generate towards a model input.

    args:
        sample: {'source': List[str], 'target': str, ...}
        tokenizer: AutoTokenizer class
        args: >= {'input_maxlen', 'decode_maxlen'}
    rets:
        features: {'input_ids', 'attention_mask', 'labels'}
    """
    
    cls_id = special_tokens_map[args.expr_name]['cls']
    sep_id = special_tokens_map[args.expr_name]['sep']

    input_ids = [cls_id]
    for text_span in sample['source']:
        span_tokens = tokenizer.tokenize(text_span)
        span_token_ids = tokenizer.convert_tokens_to_ids(span_tokens)
        input_ids.extend(span_token_ids)
        input_ids.append(sep_id)
    input_ids = input_ids[:args.input_maxlen]
    attention_mask = [1 for _ in input_ids]

    while len(input_ids) < args.input_maxlen:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)

    target_inputs = tokenizer(
        text=sample['target'], 
        padding='max_length', 
        truncation=True, 
        max_length=args.decode_maxlen
    )
    sample_features = {
        'input_ids': input_ids, 
        'attention_mask': attention_mask, 
        'decoder_attention_mask': target_inputs['attention_mask'], 
        'labels': target_inputs['input_ids']
    }
    return sample_features


def get_sample_b2b(sample, tokenizer, args):
    """Tokenize a sample to generate towards a model input.

    args:
        sample: {'table_id', 'source', 'target'}
        tokenizer: AutoTokenizer class
        args: >= {'input_maxlen', 'decode_maxlen'}
    rets:
        features: {'input_ids', 'attention_mask', 'labels'}
    """
    cls_id = special_tokens_map[args.expr_name]['cls']
    sep_id = special_tokens_map[args.expr_name]['sep']

    # concatenation
    input_ids = [cls_id]
    position_ids = [0]
    for text_span in sample['source']:
        span_tokens = tokenizer.tokenize(text_span)
        span_token_ids = tokenizer.convert_tokens_to_ids(span_tokens)
        input_ids.extend(span_token_ids)
        input_ids.append(sep_id)
        position_ids.extend([i for i in range(len(span_token_ids) + 1)])
    # truncation
    input_ids = input_ids[:args.input_maxlen]
    position_ids = position_ids[:args.input_maxlen]
    attention_mask = [1 for _ in input_ids]
    # 'max_length' padding
    while len(input_ids) < args.input_maxlen:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)

    target_inputs = tokenizer(
        text=sample['target'], 
        padding='max_length', 
        truncation=True, 
        max_length=args.decode_maxlen, 
        return_tensors='pt', 
    )
    sample_features = {
        'input_ids': input_ids, 
        'attention_mask': attention_mask, 
        'position_ids': position_ids, 
        'decoder_input_ids': target_inputs['input_ids'][0], 
        'decoder_attention_mask': target_inputs['attention_mask'][0], 
        'labels': target_inputs['input_ids'][0], 
    }
    return sample_features


SamplePrepareDict = {
    't5': get_sample_naive, 
    'bart': get_sample_naive, 
    'b2b': get_sample_b2b, 
}



def get_dataset(
    expr_name: str, 
    data_files, tokenizer, args, 
    file_type='json', 
):
    # datasets.arrow_dataset.Dataset
    raw_data = load_dataset(file_type, data_files=data_files)['train'] 
    tokenized_data = raw_data.map(
        lambda sample: SamplePrepareDict[expr_name](sample, tokenizer, args)
    )
    return tokenized_data

def get_datasets(
    expr_name: str, 
    data_dict: Dict, 
    tokenizer, 
    args, 
    file_type: str = 'json'
): 
    dataset = load_dataset(file_type, data_files=data_dict)
    for split in data_dict.keys(): 
        dataset[split].map(
        lambda sample: SamplePrepareDict[expr_name](sample, tokenizer, args)
    )
    return dataset
    


# %% test

def get_testset(data_files, file_type='json'):
    test_data = load_dataset(file_type, data_files=data_files)
    testset = test_data['train']
    return testset
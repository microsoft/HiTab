"""Generation utility functions. """


import torch 
from ..utils import special_tokens_map


# %% beam generate 

def tokenize_sample_test(sample, tokenizer, args, verbose=False):
    """Tokenize on the sample source text, while testing."""

    if verbose:
        print(f"[utils >> tknz_sample] has table {sample['table_id']} & subsent [{sample['sub_sent_id']}]")

    cls_id = special_tokens_map[args.expr_name]['cls']
    sep_id = special_tokens_map[args.expr_name]['sep']

    input_ids = [cls_id]
    position_ids = [0]
    for text_span in sample['source']:
        span_tokens = tokenizer.tokenize(text_span)
        span_token_ids = tokenizer.convert_tokens_to_ids(span_tokens)
        input_ids.extend(span_token_ids)
        input_ids.append(sep_id)
        position_ids.extend([i for i in range(len(span_token_ids) + 1)])
    input_ids = input_ids[:args.input_maxlen]
    position_ids = position_ids[:args.input_maxlen]
    attention_mask = [1 for _ in input_ids]
    
    input_ids = torch.LongTensor([input_ids])
    attention_mask = torch.LongTensor([attention_mask])
    position_ids = torch.LongTensor([position_ids])
    input_features = {
        'input_ids': input_ids.to(args.device), 
        'attention_mask': attention_mask.to(args.device), 
        'position_ids': position_ids.to(args.device)
    }
    return input_features


def clear_tokens(token_list, tokenizer):
    """Clean a token sequence by remove <pad>s. 
    Skip special tokens noted as f'<{}>'.
    """
    valid_token_list = [
        token for token in token_list
        if token not in tokenizer.all_special_tokens
    ]
    return valid_token_list


def beam_generate(sample, tokenizer, model, args, verbose=False):
    """Generate outputs from a model with beam search decoding.

    args:
        sample: {'table_id', 'sub_sent_id', 'source', 'target'}
    rets:
        generation: List[str]
    """

    # generate vocab ids
    sample_features = tokenize_sample_test(sample, tokenizer, args)
    if args.expr_name == 'b2b':
        gen_ids = model.generate(
            input_ids=sample_features['input_ids'], 
            attention_mask=sample_features['attention_mask'], 
            position_ids=sample_features['position_ids'], 
            max_length=args.decode_maxlen, 
            num_beams=args.num_beams, 
            num_return_sequences=args.num_return_sequences 
        )
    else:
        gen_ids = model.generate(
            input_ids=sample_features['input_ids'], 
            attention_mask=sample_features['attention_mask'], 
            max_length=args.decode_maxlen, 
            num_beams=args.num_beams, 
            num_return_sequences=args.num_return_sequences 
        )
    if verbose == True:
        print(f'[beam_gen] has GEN-IDS with size {gen_ids.size()}')

    gen_features = dict()
    for iret, gen_ids in enumerate(gen_ids):
        gen_tokens = tokenizer.convert_ids_to_tokens(gen_ids)
        gen_tokens_clear = clear_tokens(gen_tokens, tokenizer)
        gen_sentence = tokenizer.convert_tokens_to_string(gen_tokens_clear)
        
        gen_features[iret] = {
            'ids': gen_ids, 
            'tokens': gen_tokens, 
            'tokens_clear': gen_tokens_clear, 
            'sentence': gen_sentence
        }

    return gen_features


# %% select optimal set

from ..utils import bleu_scorer, parent_scorer 


def select_prediction_set_by_bleu(
    prediction_dicts, references, return_index=False):
    """Select sequence-wise-ly from predictions the best predset against references."""
    predictions = []
    indices = []

    for sample_pred_dict, ref_list in zip(prediction_dicts, references):
        max_idx = 0
        max_score = 0.0

        for idx, d in sample_pred_dict.items():
            res = bleu_scorer.compute(
                predictions=[d['tokens_clear']], 
                references=[ref_list]
            )
            score = res['bleu']
            
            if score > max_score:
                max_idx = idx
                max_score = score

        # print(f'[utils >> select_predset] sample max score: [{max_score}]')
        predictions.append(sample_pred_dict[max_idx]['tokens_clear'])
        indices.append(max_idx)

    if return_index: return predictions, indices
    return predictions


def select_prediction_set_by_parent(prediction_dicts, references, tables, return_index=False):
    """Select sequence-wise-ly from predictions the best predset against references."""
    predictions = []
    indices = []

    for sample_pred_dict, ref_list, table in zip(prediction_dicts, references, tables):
        max_idx = 0
        max_score = 0.0

        for idx, d in sample_pred_dict.items():
            p, r, f1, all_f1 = parent_scorer(
                predictions=[d['tokens_clear']], 
                references=[ref_list], 
                tables=[table],
                return_dict=False
            )
            
            if f1 > max_score:
                max_idx = idx
                max_score = f1

        # print(f'[utils >> select_predset] sample max score: [{max_score}]')
        predictions.append(sample_pred_dict[max_idx]['tokens_clear'])
        indices.append(max_idx)

    if return_index: return predictions, indices
    return predictions



# sort / rank multiple predictions

def rank_prediction_set_by_bleu(prediction_dicts, references):  # return_scores=True
    """Rank sequence-wise-ly from predictions the best predset against references."""
    from experiment.utils.metrics import bleu_scorer

    sorted_predictions = []
    for sample_pred_dict, ref_list in zip(prediction_dicts, references):
        pred_score_pairs = []
        for idx, d in sample_pred_dict.items():
            res = bleu_scorer.compute(
                predictions=[d['tokens_clear']], 
                references=[ref_list]
            )
            pred_score_pairs.append( (idx, d['sentence'], res['bleu']) )

        pred_score_pairs = sorted(pred_score_pairs, key=lambda x: x[2])
        sorted_predictions.append(pred_score_pairs)

    return sorted_predictions
"""Evaluation. """ 

from .utils import (
    beam_generate, 
    select_prediction_set_by_bleu, 
    select_prediction_set_by_parent, 
) 
from ..utils import bleu_scorer, parent_scorer 


def eval_with_bleu(args, testset, tokenizer, model):
    """Do evaluation on the testset, when BLEU metrics is specified. """

    raw_predictions = [
        beam_generate(sample, tokenizer, model, args)
        for sample in testset
    ]

    references = [
        [tokenizer.tokenize(sample['target'])]
        for sample in testset
    ]

    pred_tokens_dict = {}
    for idx in range(args.num_return_sequences):
        pred_tokens_dict[idx] = [sample[idx]['tokens_clear'] for sample in raw_predictions]

    for idx, predictions in pred_tokens_dict.items():
        idx_results = bleu_scorer.compute(
            predictions=predictions, 
            references=references,
        )
        print(f"Idx#{idx} - BLEU: {idx_results['bleu']: .3f}")
    
    best_predictions = select_prediction_set_by_bleu(
        raw_predictions, references, bleu_scorer)
    best_results = bleu_scorer.compute(
        predictions=best_predictions, 
        references=references
    )
    print(f"BEST BLEU: {best_results['bleu']: .3f}")

    return



def eval_with_parent(args, testset, tokenizer, model):
    """Do evaluation on the testset, when BLEU metrics is specified. """

    raw_predictions = [ beam_generate(sample, tokenizer, model, args)
        for sample in testset]
    references = [ [tokenizer.tokenize(sample['target'])]
        for sample in testset]
    tokenized_tables = []
    for sample in testset:
        raw_table_parent = sample['table_parent']
        tokenized_table_parent = []
        for attr, value in raw_table_parent:
            value_tokens = tokenizer.tokenize(value)
            tokenized_table_parent.append( ([attr], value_tokens) )
        tokenized_tables.append(tokenized_table_parent)

    pred_tokens_dict = {}
    for idx in range(args.num_return_sequences):
        pred_tokens_dict[idx] = [sample[idx]['tokens_clear'] for sample in raw_predictions]

    for idx, predictions in pred_tokens_dict.items():
        (idx_p, idx_r, idx_f1, idx_all_f1) = parent_scorer(
            predictions=predictions, 
            references=references, 
            tables=tokenized_tables, 
            return_dict=False, 
        )
        print(f"Idx#{idx} - PARENT: {idx_p:.3f}, {idx_r:.3f}, {idx_f1:.3f}")
    
    best_predictions = select_prediction_set_by_parent(
        raw_predictions, references, tokenized_tables)
    (avg_p, avg_r, avg_f, all_f) = parent_scorer(
        predictions=best_predictions, 
        references=references, 
        tables=tokenized_tables, 
        return_dict=False
    )
    print(f"BEST PARENT: {avg_p: .3f}, {avg_r:.3f}, {avg_f:.3f}")
    
    return
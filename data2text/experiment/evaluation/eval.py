"""Evaluation. """ 

from .utils import beam_generate
from ..utils import bleu_scorer, parent_scorer 


def eval_with_bleu(args, testset, tokenizer, model):
    """Do evaluation on the testset, when BLEU metrics is specified. """

    raw_predictions = [
        beam_generate(sample, tokenizer, model, args)
        for sample in testset
    ]
    predictions = [sample[0]['tokens_clear'] for sample in raw_predictions]

    references = [
        [tokenizer.tokenize(sample['target'])]
        for sample in testset
    ]

    best_results = bleu_scorer.compute(
        predictions=predictions, 
        references=references
    )
    print(f"BEST BLEU: {best_results['bleu']: .3f}")

    return



def eval_with_parent(args, testset, tokenizer, model):
    """Do evaluation on the testset, when BLEU metrics is specified. """

    raw_predictions = [ beam_generate(sample, tokenizer, model, args)
        for sample in testset]
    predictions = [sample[0]['tokens_clear'] for sample in raw_predictions]
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

    (avg_p, avg_r, avg_f, all_f) = parent_scorer(
        predictions=predictions, 
        references=references, 
        tables=tokenized_tables, 
        return_dict=False
    )
    print(f"BEST PARENT: {avg_p: .3f}, {avg_r:.3f}, {avg_f:.3f}")
    
    return
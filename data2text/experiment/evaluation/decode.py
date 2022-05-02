"""Decode. """

from .utils import (
    beam_generate, 
    rank_prediction_set_by_bleu, 
    select_prediction_set_by_parent, 
) 
from ..utils import parent_scorer 


def decode_with_bleu(args, testset, tokenizer, model):
    """Decode testset and write out, when BLEU metrics is specified. """

    raw_predictions = [
        beam_generate(sample, tokenizer, model, args)
        for sample in testset
    ]

    references = [
        [tokenizer.tokenize(sample['target'])]
        for sample in testset
    ]

    ranked_predictions = rank_prediction_set_by_bleu(
        raw_predictions, references)

    with open(args.test_decode_path, 'w') as fw:
        for idx, (pred_list, ref) in enumerate(zip(ranked_predictions, references)):
            fw.write(f"#{idx}\n")
            for ii, psent, pscore in pred_list:
                fw.write(f'[{ii}: {pscore:.4f}] {psent}\n')
            fw.write(f'{ref[0]}\n\n')
    print(f'Wrote {len(ranked_predictions)} prediction & reference instances into target file: [{args.test_decode_path}]')

    return


def decode_with_parent(args, testset, tokenizer, model):
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

    with open(args.test_decode_path, 'w') as fw:
        for idx, (pred, ref, tab) in enumerate(zip(best_predictions, references, tokenized_tables)):
            sample_parent = parent_scorer(
                predictions=[pred], 
                refereces=[ref], 
                tables=[tab], 
                return_dict=True
            )
            fw.write(f"#{idx} BLEU: [{sample_parent['average_f1']:.4f}]\n")
            fw.write(f'{pred}\n{ref[0]}\n\n')
    print(f'Wrote {len(predictions)} prediction & reference pairs into target file: [{args.test_decode_path}]')

    return
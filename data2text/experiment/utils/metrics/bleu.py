"""BLEU(-4 by default) evaluation metric. 
 - bleu_scorer: input `predictions` and list of `references` to calculate scores. 
 - bleu_metric_builder: a function that performs evaluation with paired tokenizer. 
""" 


from datasets import load_metric

bleu_scorer = load_metric('bleu')


def bleu_metric_builder(tokenizer, bleu_scorer=bleu_scorer):
    """A builder of the BLEU Metrics."""

    def compute_bleu_metrics(pred, verbose=False):
        """utility to compute BLEU during training."""
        labels_ids = pred.label_ids
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        label_tokens = [[tokenizer.tokenize(str)] for str in label_str]   # multiple lists of tokens for each sample
        
        pred_ids = pred.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        pred_tokens = [tokenizer.tokenize(str) for str in pred_str]

        # compute the metric.
        # ['bleu', 'precisions', 'brevity_penalty', 'length_ratio', 'translation_length', 'reference_length']
        bleu_results = bleu_scorer.compute(
            predictions=pred_tokens,
            references=label_tokens, 
            smooth=False
        )

        if verbose == True:
            print(f'\n\nBLEU Results:')
            print(f"bleu: {bleu_results['bleu']:.4f}")
            print(f"precisions: {[round(item,4) for item in bleu_results['precisions']]}")
            print(f"brevity_penalty: {bleu_results['brevity_penalty']:.4f}")
            print(f"length_ratio: {bleu_results['length_ratio']:.4f}")
            print(f"translation_length: {bleu_results['translation_length']}")
            print(f"reference_length: {bleu_results['reference_length']}\n\n")
        
        return {'bleu-4': round(bleu_results['bleu'], 4)}

    return compute_bleu_metrics
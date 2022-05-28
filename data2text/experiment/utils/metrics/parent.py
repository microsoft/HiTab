"""Parent evaluation metrics. 
That additionally take tables into consideration. 
 - parent_scorer 
 - parent_metric_builder 
""" 

import math
import collections


# %% utility functions 

def overlap_probability(ngram, table, smoothing=0.0, stopwords=None):
    """Returns the probability that the given n-gram overlaps with the table."""
    # pylint: disable=g-complex-comprehension
    if len(table[0]) == 2:
        table_values = set([tok for _, value in table for tok in value])
    else:
        table_values = set([tok for head, _, tail in table for tok in head + tail])

    overlap = 0
    for token in ngram:
        if stopwords is not None and token in stopwords:
            overlap += 1
            continue
        if token in table_values:
            overlap += 1
    return float(overlap + smoothing) / float(len(ngram) + smoothing)    

def _lcs(x, y):
    """Computes the length of the LCS between two seqs. """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table

def _len_lcs(x, y):
    """Returns the length of the Longest Common Subsequence between two seqs. """
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]

def _mention_probability(table_entry, sentence, smoothing=1e-6):   # smoothing=0.0
    """Returns the probability that the table entry is mentioned in the sentence."""
    if len(table_entry) == 2:
        value = table_entry[1]
    else:
        value = table_entry[0] + table_entry[2]
    overlap = _len_lcs(value, sentence)
    return float(overlap + smoothing) / float(len(value) + smoothing)    

def _ngrams(sequence, order):
    """Yields all ngrams of given order in sequence."""
    assert order >= 1
    for n in range(order, len(sequence) + 1):
        yield tuple(sequence[n - order: n])

def _ngram_counts(sequence, order):
    """Returns count of all ngrams of given order in sequence."""
    if len(sequence) < order:
        return collections.Counter()
    return collections.Counter(_ngrams(sequence, order))



# %% metrics calculation and builder 

def parent_scorer(
    predictions, references, tables, 
    lambda_weight=0.5, smoothing=1e-5, max_order=4, 
    entailment_fn=overlap_probability, mention_fn=_mention_probability, 
    return_dict=True
):
    """Metric for comparing predictions to references given tables.
    args: 
        predictions: List[str] 
        references: List[ List[str] ]
        tables: List[ List[Tuple(str-field, str-cell)] ]
        ... 
    rets: 
        result: Dict{
            'average_precision', 'average_recall', 
            'average_f1', 'all_f1_scores'
        }
    """
    
    precisions, recalls, all_f_scores = [], [], []
    reference_recalls, table_recalls = [], []
    all_lambdas = []

    for prediction, list_of_references, table in zip(predictions, references, tables):
        c_prec, c_rec, c_f = [], [], []
        ref_rec, table_rec = [], []

        for reference in list_of_references:
            # Weighted ngram precisions and recalls for each order.
            ngram_prec, ngram_rec = [], []
            for order in range(1, max_order + 1):
                # Collect n-grams and their entailment probabilities.
                pred_ngram_counts = _ngram_counts(prediction, order)
                pred_ngram_weights = {ngram: entailment_fn(ngram, table)
                                    for ngram in pred_ngram_counts}
                ref_ngram_counts = _ngram_counts(reference, order)
                ref_ngram_weights = {ngram: entailment_fn(ngram, table)
                                    for ngram in ref_ngram_counts}

                # Precision.
                numerator, denominator = 0., 0.
                for ngram, count in pred_ngram_counts.items():
                    denominator += count
                    prob_ngram_in_ref = min(
                        1., float(ref_ngram_counts.get(ngram, 0) / count))
                    numerator += count * (
                        prob_ngram_in_ref +
                        (1. - prob_ngram_in_ref) * pred_ngram_weights[ngram])
                if denominator == 0.:
                    # Set precision to 0.
                    ngram_prec.append(0.0)
                else:
                    ngram_prec.append(numerator / denominator)

                # Recall.
                numerator, denominator = 0., 0.
                for ngram, count in ref_ngram_counts.items():
                    prob_ngram_in_pred = min(
                        1., float(pred_ngram_counts.get(ngram, 0) / count))
                    denominator += count * ref_ngram_weights[ngram]
                    numerator += count * ref_ngram_weights[ngram] * prob_ngram_in_pred
                if denominator == 0.:
                    # Set recall to 1.
                    ngram_rec.append(1.0)
                else:
                    ngram_rec.append(numerator / denominator)

            # Compute recall against table fields.
            table_mention_probs = [mention_fn(entry, prediction) for entry in table]
            table_rec.append(sum(table_mention_probs) / len(table))

            # Smoothing.
            for order in range(1, max_order):
                if ngram_prec[order] == 0.:
                    ngram_prec[order] = smoothing
                if ngram_rec[order] == 0.:
                    ngram_rec[order] = smoothing

            # Compute geometric averages of precision and recall for all orders.
            w = 1. / max_order
            if any(prec == 0. for prec in ngram_prec):
                c_prec.append(0.)
            else:
                sp = (w * math.log(p_i) for p_i in ngram_prec)
                c_prec.append(math.exp(math.fsum(sp)))
            if any(rec == 0. for rec in ngram_rec):
                ref_rec.append(smoothing)
            else:
                sr = [w * math.log(r_i) for r_i in ngram_rec]
                ref_rec.append(math.exp(math.fsum(sr)))

            # Combine reference and table recalls.
            if table_rec[-1] == 0.:
                table_rec[-1] = smoothing
            if ref_rec[-1] == 0. or table_rec[-1] == 0.:
                c_rec.append(0.)
            else:
                if lambda_weight is None:
                    lw = sum([mention_fn(entry, reference) for entry in table]) / len(table)
                    lw = 1. - lw
                else:
                    lw = lambda_weight
                all_lambdas.append(lw)
                c_rec.append(
                    math.exp((1. - lw) * math.log(ref_rec[-1]) +
                            (lw) * math.log(table_rec[-1])))
            
            # F-score.
            c_f.append((2. * c_prec[-1] * c_rec[-1]) / (c_prec[-1] + c_rec[-1] + 1e-8))

        # Get index of best F-score.
        max_i = max(enumerate(c_f), key=lambda x: x[1])[0]
        precisions.append(c_prec[max_i])
        recalls.append(c_rec[max_i])
        all_f_scores.append(c_f[max_i])
        reference_recalls.append(ref_rec[max_i])
        table_recalls.append(table_rec[max_i])

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f_score = sum(all_f_scores) / len(all_f_scores)

    if return_dict: 
        result_dict = {
            'average_precision': avg_precision, 
            'average_recall': avg_recall, 
            'average_f1': avg_f_score, 
            'all_f1_scores': all_f_scores
        }
        return result_dict
    else:
        return avg_precision, avg_recall, avg_f_score, all_f_scores


def parent_metric_builder(tokenizer, parent_scorer=parent_scorer):
    """A builder of the PARENT metrics given a compatible tokenizer. """

    def compute_parent_metrics(pred, tables, verbose=False):
        labels_ids = pred.label_ids
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        label_tokens = [[tokenizer.tokenize(str)] for str in label_str] 

        pred_ids = pred.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        pred_tokens = [tokenizer.tokenize(str) for str in pred_str]

        parent_results = parent_scorer(
            predictions=pred_tokens,
            references=label_tokens, 
            tables=tables, 
            return_dict=True
        )
        
        if verbose == True:
            n = len(parent_results['all_f1_scores'])
            p = parent_results['average_precision']
            r = parent_results['average_recall']
            f = parent_results['average_f1']
            print(f'[{n} instances]: avg precision: {p:.3f}, recall: {r:.3f}, f1: {f:.3f}')
        return parent_results
    
    return compute_parent_metrics
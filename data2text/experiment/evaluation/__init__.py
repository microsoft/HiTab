"""Quick imports of Evaluation Modules. """ 

from .eval import eval_with_bleu, eval_with_parent 

EvalDict = {
    'bleu': eval_with_bleu, 
    'parent': eval_with_parent, 
}

from .decode import decode_with_bleu, decode_with_parent
DecodeDict = {
    'bleu': decode_with_bleu, 
    'parent': decode_with_parent, 
}
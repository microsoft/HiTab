"""Test with a specified evaluation metrics. """


import os 
from .utils import prepare_tokenizer, get_testset, ModelTestDict 
from .pointer_generator import BeamSearch 
from .evaluation import EvalDict, DecodeDict 



def run_test(args): 
    """Test a fine-tuned model. Using huggingface/transformers. 
    Load the test set, prepare tokenizer, load tuned models. 
    Then perform evaluation or decoding. 
    """
    testset = get_testset(data_files=args.test_outpath)
    tokenizer = prepare_tokenizer(name=args.tokenizer_name)

    model = ModelTestDict[args.expr_name](
        run_dir=args.run_dir, 
        path=args.model_path, 
        name=args.model_name, 
        device=args.device, 
    )
    if args.do_test: 
        for metric in args.metrics: 
            print(f'Start evaluation with metrics [{metric}]') 
            EvalDict[metric](args, testset, tokenizer, model)
    if args.do_decode:
        args.test_decode_path = os.path.join(args.run_dir, args.test_decode_name)
        DecodeDict[args.metrics[0]](args, testset, tokenizer, model)



# pointer generator network 

def find_best_pgn_model_index(run_dir, main_metric_key='bleu-4'):
    """Find the best model at testing. """
    detailed_run_dir = os.path.join(run_dir, 'train', 'models')
    decode_dirs = os.listdir(detailed_run_dir)
    decode_metrics = []
    for dd in decode_dirs:
        mfile = os.path.join(run_dir, dd, 'metrics')
        ckpt_metrics = {}
        with open(mfile, 'r') as fr:
            for line in fr:
                mkey, mval = line.strip().split('\t')
                ckpt_metrics[mkey] = float(mval)
        decode_metrics.append(ckpt_metrics)
    
    best_ckpt_idx = -1
    best_ckpt_mval = 0.0
    for idx, mdict in decode_metrics:
        mval = mdict[main_metric_key]
        if mval > best_ckpt_mval:
            best_ckpt_mval = mval
            best_ckpt_idx = idx
    return best_ckpt_idx
    

def run_test_pgn(args):
    try:
        # best_ckpt_idx = find_best_pgn_model_index(args.run_dir)
        best_ckpt_idx = 99
        best_ckpt_path = os.path.join(args.run_dir, 'train', 'models', f'model_{best_ckpt_idx}')
    except:
        best_ckpt_path = None
    print(f'<<< Perform the Final Test ... (use model [{best_ckpt_path}]) >>>')
    tester = BeamSearch(args, best_ckpt_path, args.decode_data_path)
    tester.run(args.logging_steps)
    # tester.eval_parent(args.logging_steps)
    print(f'<<< Finished the Final Test ! >>>')



# collection

TestFunctionDict = {
    't5': run_test, 
    'bart': run_test, 
    'b2b': run_test, 
    'pg': run_test_pgn, 
}
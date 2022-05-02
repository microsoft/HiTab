"""Train & Test Pipeline for HiTab Data-to-Text Generation. 

Available Models: 
 - (t5) T5: base size by default 
 - (bart) BART: base size by default 
 - (b2b) BERT-to-BERT: base size by default 
 - (pgn) Pointer-Generator Network 

"""

import os 
import argparse 

from experiment.train_d2t import TrainFunctionDict 
from experiment.eval_d2t import TestFunctionDict 


def main(): 
    if args.do_train or args.do_eval: 
        TrainFunctionDict[args.expr_name](args) 
    
    if args.do_test or args.do_decode: 
        TestFunctionDict[args.expr_name](args) 



ExperimentSuite = {
    't5': {
        'model_name': 't5-large', 
        'tokenizer_name': 't5-large', 
        'per_device_train_batch_size': 2, 
        'per_device_eval_batch_size': 2, 
        'learning_rate': 1e-4, 
        'num_train_epochs': 50, 
    }, 
    'bart': {
        'model_name': 'facebook/bart-base', 
        'tokenizer_name': 'facebook/bart-base', 
        'per_device_train_batch_size': 8, 
        'per_device_eval_batch_size': 8, 
        'learning_rate': 1e-4, 
        'num_train_epochs': 50, 
    }, 
    'b2b': {
        'model_name': 'bert-large-uncased', 
        'tokenizer_name': 'bert-large-uncased', 
        'per_device_train_batch_size': 8, 
        'per_device_eval_batch_size': 8, 
        'learning_rate': 1e-4, 
        'num_train_epochs': 50, 
    }, 
    'pgn': {
        'model_name': None, 
        'tokenizer_name': None, 
        'per_device_train_batch_size': 2, 
        'per_device_eval_batch_size': 2, 
        'learning_rate': 1e-3, 
        'num_train_epochs': 100, 
        'train_sleep_time': 15, 
        'vocab_path': os.path.join(os.getcwd(), 'experiment/pointer_generator/vocab'), 
        'vocab_size': 30000, 
        'test_decode_name': 'decoded_test.log', 
    }
}



def update_arguments(): 
    # misc 
    args.run_dir = os.path.join(os.getcwd(), args.run_subdir, args.expr_name) 
    if not os.path.exists(args.run_dir): os.makedirs(args.run_dir) 

    # data 
    args.train_outpath = args.train_data_path = os.path.join(args.data_dir, 'train_samples.jsonl')
    args.dev_outpath = args.eval_data_path = os.path.join(args.data_dir, 'dev_samples.jsonl')
    args.test_outpath = args.decode_data_path = os.path.join(args.data_dir, 'test_samples.jsonl')

    # model 
    for k, v in ExperimentSuite[args.expr_name].items(): 
        setattr(args, k, v)
    args.latest_model_path = args.model_path 

    return args 



if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 

    # data 
    parser.add_argument('--data_dir', type=str, default='data', 
        help="Directory containing the processed train/dev/test_samples.jsonl files.")
    
    # model 
    parser.add_argument('--expr_name', type=str, default='t5', 
        choices=['t5','bart','b2b','pgn'], help="Model name (abbr.).")
    parser.add_argument('--model_path', type=str, default=None, 
        help="Path of model checkpoint if used for weight initialization.")

    # training 
    parser.add_argument('--logging_steps', type=int, default=100) 
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    parser.add_argument('--start_iepoch', type=int, default=0, 
        help='Index of the starting epoch.')
    parser.add_argument('--num_train_epochs', type=int, default=5, 
        help='Number of epochs for continual tuning.')
    parser.add_argument('--num_eval_epochs', type=int, default=1, 
        help='Number of epochs per validation.')
    parser.add_argument('--num_save_model_epochs', type=int, default=1, 
        help='Number of epochs to save model ckpt.')

    parser.add_argument('--input_maxlen', type=int, default=512, 
        help='Max number of tokens of input sequences.')
    parser.add_argument('--decode_maxlen', type=int, default=100, 
        help='Max number of tokens of generated sequnces.')
    parser.add_argument('--num_beams', type=int, default=5, 
        help='Number of the searching beam size for sequence generation.')
    parser.add_argument('--num_return_sequences', type=int, default=3, 
        help='Number of generated sentences for comparison.')

    # evaluation
    parser.add_argument('--metrics', type=str, nargs='+', default=['bleu'])

    # misc 
    parser.add_argument('--run_subdir', type=str, default='runs')
    parser.add_argument('--log_subdir', type=str, default='logs') 

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=47)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_test', action='store_true') 
    parser.add_argument('--do_decode', action='store_true')
    
    args = parser.parse_args() 

    args = update_arguments()
    main() 
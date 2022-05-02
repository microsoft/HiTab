"""Training Functions. 

 - `run_train` for T5, BART, and BERT-to-BERT (huggingface/transformers supported)
 - `run_train_pgn` for Pointer-Generator Network 
"""

import os
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from .utils import (
    prepare_tokenizer, get_dataset, 
    ModelPrepareDict, MetricsBuildDict
)
from .pointer_generator.trainer import Trainer as PgnTrainer
from .pointer_generator.decode import BeamSearch


# %% huggingface-supported models: t5, bart, bert-to-bert

def prepare_training_arguments(args):
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.run_dir, 
        do_train=args.do_train,
        do_eval=args.do_eval,
        evaluation_strategy='epoch', 
        save_strategy='epoch', 
        logging_steps=args.logging_steps,   
        # optimization args, the trainer uses the Adam optimizer 
        # and has a linear warmup for the learning rates
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, 
        num_train_epochs=args.num_train_epochs, 
        warmup_steps=args.warmup_steps, 
        # misc args
        seed=args.seed, 
        disable_tqdm=False, 
        load_best_model_at_end=True, 
        metric_for_best_model='bleu-4', 
        # generation
        predict_with_generate=True
    )
    return train_args


def run_train(args):
    """A general training script with huggingface/transformers. 
    1. prepare training and validation sets 
    2. load model and organize trainer (and arguments) 
    """ 
    tokenizer = prepare_tokenizer(name=args.tokenizer_name)
    trainset= get_dataset(
        expr_name=args.expr_name, 
        data_files=args.train_outpath, 
        tokenizer=tokenizer, 
        args=args
    )
    validset = get_dataset(
        expr_name=args.expr_name, 
        data_files=args.dev_outpath, 
        tokenizer=tokenizer, 
        args=args
    )

    train_args = prepare_training_arguments(args)
    model = ModelPrepareDict[args.expr_name](
        name=args.model_name, 
        path=args.model_path, 
        device=args.device
    )

    metric_fn = MetricsBuildDict[args.metrics[0]](tokenizer)
    trainer = Seq2SeqTrainer(
        model=model, 
        args=train_args, 
        train_dataset=trainset, eval_dataset=validset, 
        tokenizer=tokenizer, compute_metrics=metric_fn, 
    )

    trainer._max_length = args.decode_maxlen
    trainer._num_beams = args.num_beams

    trainer.train()



# %% pointer-generator network

def find_latest_pgn_model_path(model_dir):
    """Find the path/filename of the latest model within the given directory."""
    filenames = os.listdir(model_dir)
    if len(filenames) == 0: return 

    indices = []
    for fn in filenames:
        model_name = fn.split('.')[0]
        model_index = int(model_name.split('_')[-1])
        indices.append(model_index)
    max_index = indices.index( max(indices) )
    max_file = filenames[max_index]
    
    latest_model_path = os.path.join(model_dir, max_file)
    return latest_model_path



def run_train_pgn(args):
    trainer = PgnTrainer(args)
    if args.latest_model_path is not None:
        model_path = args.latest_model_path
    else:
        model_path = args.model_path
    print(f'run with model from [{model_path}]')

    for iepoch in range(args.start_iepoch, args.start_iepoch + args.num_train_epochs):
        print(f'\n <<< START of the #{iepoch} EPOCH >>>')
        if (iepoch + 1) % args.num_eval_epochs == 0: 
            do_eval = True
        else: 
            do_eval = False 

        if (iepoch + 1) % args.num_save_model_epochs == 0: 
            do_save_model = True
        else: 
            do_save_model = False 

        trainer.run_one_epoch(
            iepoch=iepoch, 
            model_path=model_path, 
            interval=args.logging_steps, 
            save_model=do_save_model, 
        )
        args.latest_model_path = find_latest_pgn_model_path(trainer.model_dir) 

        if (do_eval == True) and (args.latest_model_path is not None):
            print(f'EVAL using model [{args.latest_model_path}]')
            tester = BeamSearch(args, args.latest_model_path, args.eval_data_path)
            tester.run(args.logging_steps)
        print(f' <<< END of the #{iepoch} EPOCH >>>\n')
    




# %% collection

TrainFunctionDict = {
    't5': run_train, 
    'bart': run_train, 
    'b2b': run_train, 
    'pgn': run_train_pgn, 
} 
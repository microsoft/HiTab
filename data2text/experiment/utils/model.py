"""Build or Load, Pre-trained or Tuned Models."""


import os
import json
from transformers import (
    AutoModelForSeq2SeqLM, BertGenerationEncoder, 
    BertGenerationDecoder, EncoderDecoderModel, 
)

import logging 
logger = logging.getLogger(__name__)

# %% train 

def prepare_model_naive(name: str, path: str, device: str = 'cuda'):
    """Load target model from the specified name or model-file path."""
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
        return model.to(device)
    except:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(path)
            return model.to(device)
        except:
            logger.error(f'[utils >> prep_model] fails with name [{name}] and path [{path}]')
    

def prepare_b2b_model(name: str, path: str, device: str = 'cuda'):
    """Prepare a EncoderDecoderModel class from BertGenerationEncoder + BertGenerationDecoder."""
    if path is not None:
        bert2bert = EncoderDecoderModel.from_pretrained(path)
    elif name is not None:
        encoder = BertGenerationEncoder.from_pretrained(
            name, bos_token_id=101, eos_token_id=102)
        decoder = BertGenerationDecoder.from_pretrained(
            name, bos_token_id=101, eos_token_id=102, 
            add_cross_attention=True, is_decoder=True)
        bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)

        # adjust default configs
        bert2bert.config.encoder.max_length = 512
        bert2bert.config.decoder.max_length = 60
    return bert2bert.to(device)



ModelPrepareDict = {
    't5': prepare_model_naive, 
    'bart': prepare_model_naive, 
    'b2b': prepare_b2b_model
}


# %% test

def find_best_model(run_dir, load_last: bool = True):
    if run_dir is None: return None
    model_ckpts = [rd for rd in os.listdir(run_dir) if rd.startswith('checkpoint')]
    if len(model_ckpts) == 0: return None

    print(f"RUN-DIR: {run_dir}")
    print(f"MODEL-CKPTS: {model_ckpts}")
    
    iters = [int(dirname.split('-')[-1]) for dirname in model_ckpts]
    index = iters.index( max(iters) )
    model_path = os.path.join(run_dir, model_ckpts[index])
    if load_last: return model_path
    

    trainer_state_file = os.path.join(model_path, 'trainer_state.json')
    with open(trainer_state_file, 'r') as fr:
        states = json.load(fr)
    best_model_path = states['best_model_checkpoint']
    return best_model_path


def load_model_test_naive(run_dir, path, name, device):
    """Load model from 1) the running directory, 2) specified path, 3) library model name."""
    
    best_model_path = find_best_model(run_dir)
    if best_model_path is not None:
        logging.info(f'[utils >> load_model] from tuned checkpoint [{best_model_path}]')
        model = AutoModelForSeq2SeqLM.from_pretrained(best_model_path)
        return model.to(device)

    logging.info(f'[utils >> load_model] fails import from run-dir [{run_dir}]')
    try:
        model_path = path
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        logging.info(f'[utils >> load_model] from original path [{model_path}]')
        return model.to(device)
    except:
        logging.warning(f'[utils >> load_model] fails import from path [{path}]')
    
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
        logging.info(f'[utils >> load_model] from name [{name}]')
        return model.to(device)
    except:
        logging.warning(f'[utils >> load_model] fails import from name [{name}]')
    
    return None


def load_model_test_b2b(run_dir, path, name, device):
    """Load model from 1) the running directory, 2) specified path, 3) library model name."""

    best_model_path = find_best_model(run_dir)
    if best_model_path is not None:
        logging.info(f'[utils >> load_model] from tuned checkpoint [{best_model_path}]')
        model = EncoderDecoderModel.from_pretrained(best_model_path)
        return model.to(device)

    logging.info(f'[utils >> load_model] fails import from run-dir [{run_dir}]')
    try:
        model_path = path
        model = EncoderDecoderModel.from_pretrained(model_path)
        logging.info(f'[utils >> load_model] from original path [{model_path}]')
        return model.to(device)
    except:
        logging.warning(f'[utils >> load_model] fails import from path [{path}]')
    
    try:
        encoder = BertGenerationEncoder.from_pretrained(name)
        decoder = BertGenerationDecoder.from_pretrained(name)
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
        logging.info(f'[utils >> load_model] from name [{name}]')
        return model.to(device)
    except:
        logging.warning(f'[utils >> load_model] fails import from name [{name}]')
    
    return None



ModelTestDict = {
    't5': load_model_test_naive, 
    'bart': load_model_test_naive, 
    'b2b': load_model_test_b2b
}
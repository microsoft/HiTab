"""Overall Model Architecture."""

import torch
from experiment.pointer_generator.model.layers import Encoder, ReduceState, Decoder



class Model(object):
    """Model class consists of an encoder, a reduce-state, and a decoder."""

    def __init__(self, config, model_path=None, is_eval=False, is_transformer=False):
        super(Model, self).__init__()
        
        encoder = Encoder(config)
        decoder = Decoder(config)
        reduce_state = ReduceState(config)
        if is_transformer:
            print(f'Transformer Encoder is not yet available.')
        
        # share the embedding between encoder and decoder
        decoder.tgt_word_emb.weight = encoder.src_word_emb.weight

        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()
        
        self.use_cuda = config.use_gpu and torch.cuda.is_available()
        if self.use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()
        
        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_path is not None:
            state = torch.load(model_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])
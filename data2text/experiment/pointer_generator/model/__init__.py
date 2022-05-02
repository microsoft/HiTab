"""Initialize the suite of pointer-generator network models. 

With a 'Basic Module' that supports model-weight initializations.
"""

import math
import torch.nn as nn



# %% Basic Module for models

class BasicModule(nn.Module):
    """Initialization for models."""

    def __init__(self, init_method='uniform'):
        """Initialize model weights with uniform distribution by default.
        init_method: choices = ['uniform', 'normal']
        future-to-add: 'truncated_normal'
        """
        super(BasicModule, self).__init__()
        self.init_method = init_method

    def init_params(self, init_range=0.05):
        """Initialize self weights/parameters with the specified method."""

        if not (self.init_method in ['uniform', 'normal']):    # 'truncated_normal'
            print(f'[BasicModule >> init_params] not supporting init_method: {self.init_method}')
            return

        for param in self.parameters():
            if (not param.requires_grad) or len(param.shape) == 0: 
                continue
            if self.init_method == 'uniform':
                nn.init.uniform_(param, a=-init_range, b=init_range)
            else:
                stddev = 1 / math.sqrt(param.shape[0])
                if self.init_method == 'normal':
                    nn.init.normal_(param, std=stddev)
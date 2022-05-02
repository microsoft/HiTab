"""Quick imports of all evaluation metrics. 

BLEU: require 'source' and 'target' text tokens. 
PARENT: require 'source' & 'target' tokens, and 'table_parent' list of tuples. 

"""

from .bleu import bleu_metric_builder, bleu_scorer  
from .parent import parent_metric_builder, parent_scorer  

MetricsDict = {
    'bleu': bleu_scorer, 
    'parent': parent_scorer, 
}

MetricsBuildDict = {
    'bleu': bleu_metric_builder, 
    'parent': parent_metric_builder, 
} 

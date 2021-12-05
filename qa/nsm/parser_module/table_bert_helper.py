from typing import Dict, List
import json

from qa.table_bert.table_bert import TableBertModel


def get_table_bert_model(config: Dict, use_proxy=False, master=None):
    model_name_or_path = config.get('table_bert_model_or_config')
    if model_name_or_path in {None, ''}:
        model_name_or_path = config.get('table_bert_config_file')
    if model_name_or_path in {None, ''}:
        model_name_or_path = config.get('table_bert_model')

    table_bert_extra_config = config.get('table_bert_extra_config', dict())

    # print(f'Loading table BERT model {model_name_or_path}', file=sys.stderr)
    model = TableBertModel.from_pretrained(
        model_name_or_path,
        **table_bert_extra_config
    )

    print('Table Bert Config')
    print(json.dumps(vars(model.config), indent=2))

    return model


def get_table_bert_input_from_context(
    env_context: List[Dict],
):
    contexts = []
    tables = []

    for e in env_context:
        contexts.append(e['question_tokens'])
        tables.append(e['table'])

    return contexts, tables

# -*- coding: utf-8 -*-
from codecs import open
from argparse import ArgumentParser
from stanza.server import CoreNLPClient
from nltk.stem import SnowballStemmer
import nltk

from qa.datadump.utils import *


n_total_num = 0
n_filtered_num = 0
n_date_and_num = 0
n_too_few_num_date = 0


def table2kg(table_info):
    """ Convert input .json table to kg following tabert implementation, actually NOT KG in our hierarchical tables."""
    props, num_props, datetime_props = find_props(table_info)
    return dict(kg=table_info, props=props, num_props=num_props, datetime_props=datetime_props)


# Don't use dataframe.from_csv because quotes might be dropped.
def create_df_from_questions(fn):
    df_dict = {}
    with open(fn, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quotechar=None)
        header = next(reader)
        for col in header:
            df_dict[col] = []
        for line in reader:
            for col, val in zip(header, line):
                df_dict[col].append(val)
    df = pandas.DataFrame(df_dict)
    return df


def create_dict_from_questions(fn):
    dicts = {}
    with open(fn, 'r') as f:
        for line in f:
            data = json.loads(line)
            dicts[data['id']] = data
    return dicts


def tokens_contain(string_1, string_2):
    tks_1 = nltk.tokenize.word_tokenize(string_1)
    tks_2 = nltk.tokenize.word_tokenize(string_2)
    # tks_1 = string_1.split()
    # tks_2 = string_2.split()
    return set(tks_2).issubset(set(tks_1))


def string_in_table_tk(string, table):
    """ Check if a tokenized string is a subset of table string tokens. Modified to adapt to hmt."""
    top_root = table['kg']['top_root']
    left_root = table['kg']['left_root']
    if string in ['<LEFT>', '<TOP>']:
        return False
    return _string_in_table_tk(string, top_root) or _string_in_table_tk(string, left_root)


def _string_in_table_tk(string, node):
    """ DFS."""
    if tokens_contain(node['name'], string):
        return True
    # if 'children_dicts' not in node:
    #     return False
    # for child in node['children_dicts']:
    #     if _string_in_table_tk(string, child):
    #         return True
    if 'children_dict' not in node:
        return False
    for child in node['children_dict']:
        if _string_in_table_tk(string, child):
            return True
    return False


def string_in_table_str(string, table):  # 只要string是某个cell子串('app也是apple的子串')，就认为是entity。
    """ Check if a string is in table string. Modified to adapt to hmt."""
    top_root = table['kg']['top_root']
    left_root = table['kg']['left_root']
    if string in ['<LEFT>', '<TOP>']:
        return False
    return _string_in_table_str(string, top_root) or _string_in_table_str(string, left_root)


def _string_in_table_str(string, node):
    """ DFS."""
    if string in node['name']:
        return True
    # if 'children_dicts' not in node:
    #     return False
    # for child in node['children_dicts']:
    #     if _string_in_table_str(string, child):
    #         return True
    if 'children_dict' not in node:
        return False
    for child in node['children_dict']:
        if _string_in_table_str(string, child):
            return True
    return False


def string_in_table(string, kg, use_tokens_contain):
    if use_tokens_contain:
        return string_in_table_tk(string, kg)
    else:
        return string_in_table_str(string, kg)


def date_in_table(ent, table):  # TODO: not supported, and value of utterance/table datetime is not in same format.
    props_set = set(table['datetime_props'])
    for k, node in table['kg'].items():
        for prop, val in node.items():
            if prop in props_set and ent == val:
                return True
    else:
        return False


def num_in_table(ent, table):  # TODO: not supported
    props_set = set(table['num_props'])
    for k, node in table['kg'].items():
        for prop, val in node.items():
            if prop in props_set and ent == val:
                return True
    else:
        return False


def prop_in_question_score(prop, question, stop_words, binary=True):
    """ Check if a header is in question. Modified to adapt to hmt."""
    # prop = prop[2:]
    # prop = u'-'.join(prop.split(u'-')[:-1])
    prop_tks = prop.split(u'_')
    n_in_question = 0
    for tk in prop_tks:
        tk = tk
        if tk not in stop_words and tk in question:
            n_in_question += 1
    if binary:
        n_in_question = min(n_in_question, 1)
    return n_in_question


def n_gram(tks, n):
    """ Return token-level n-gram of given tokens."""
    return [set(tks[i: i + n]) for i in range(len(tks) - n + 1)]


def prop_in_question_score_n_gram(prop, question, stop_words, binary=True, binary_threshold=0.45,
                                  stemmer=SnowballStemmer('english')):
    """ Check if a header is in question."""
    prop_tks = [normalize(tk) for tk in nltk.word_tokenize(prop) if tk not in stop_words]
    question_tks = [normalize(tk) for tk in nltk.word_tokenize(question) if tk not in stop_words]
    if stemmer is not None:
        prop_tks = [stemmer.stem(tk) for tk in prop_tks]
        question_tks = [stemmer.stem(tk) for tk in question_tks]

    max_n_gram_length = min(len(prop_tks), len(question_tks))
    if max_n_gram_length == 0:
        return 0

    score = 0
    prop_tks_set = set(prop_tks)
    for i in range(1, max_n_gram_length + 1):
        question_n_grams = n_gram(question_tks, i)
        for question_n_gram in question_n_grams:
            if question_n_gram.issubset(prop_tks_set):
                score = max(score, i / len(prop_tks_set))
                break

    if binary:
        score = 1 if score > binary_threshold else 0
    return score


def find_props(table_info):
    """ Find all headers in kg."""  # TODO: currently entity not including index name.
    top_root = table_info['top_root']
    left_root = table_info['left_root']
    props, num_props, datetime_props = set(), set(), set()
    iter_nodes(top_root, props, num_props, datetime_props)
    iter_nodes(left_root, props, num_props, datetime_props)
    return list(props), list(num_props), list(datetime_props)


def iter_nodes(node, props, num_props, datetime_props):
    """ DFS."""
    if node['name'] not in ['<LEFT>', '<TOP>']:
        props.add(node['name'])
        if node['type'] == 'number':  # TODO: currently all 'string'
            num_props.add(node['name'])
        if node['type'] == 'datetime':
            datetime_props.add(node['name'])
    if 'children_dict' not in node:
        return
    else:
        for child in node['children_dict']:
            iter_nodes(child, props, num_props, datetime_props)


def collect_examples_from_dict(args, sample_dict, table_dict, stop_words):
    """ Construct MAPO input examples."""
    example_linked_list = []
    for sample_id, sample in sample_dict.items():
        context = sample['table_id']
        table = table_dict[sample['table_id']]
        label_sample_with_corenlp(sample, args.client)
        tks = sample['tokens'].split('|')
        pos_tags = sample['pos_tags'].split('|')
        vals = sample['ner_vals'].split('|')
        tags = sample['ner_tags'].split('|')
        answer = sample['answer']
        e = dict(id=sample['id'], context=context,
                 question=sample['question'], answer=answer,
                 tokens=tks, pos_tags=pos_tags)
        # entities are normalized tokens
        e['entities'] = []
        e['processed_tokens'] = tks[:]
        e['in_table'] = [0] * len(tks)
        # some annotations beneficial for partial supervision
        e['linked_cells'] = sample['linked_cells']
        e['aggregation'] = sample['aggregation']

        for i, (tk, tag, val) in enumerate(zip(tks, tags, vals)):
            if tk not in stop_words:  # 判断每个tk是否为cell value的一部分，True即为entity，并不是传统意义的entity。
                if string_in_table(normalize(tk), table, args.use_tokens_contain):
                    e['entities'].append(  # entity会被加入memory中，暂时没有用
                        dict(value=[normalize(tk)], token_start=i, token_end=i + 1,
                             type='string_list'))
                    e['in_table'][i] = 1
            if tag != 'O':
                e['processed_tokens'][i] = '<{}>'.format(tag)
        e['features'] = [[it] for it in e['in_table']]  # features即当前tk是否在table中

        prop_features = dict(  # 若第i个prop有tk出现在question中，则prop_features[prop]=1
            [(prop, [prop_in_question_score(
                prop, e['question'], stop_words,
                binary=not args.use_prop_match_count_feature)])
             for prop in table['props']])
        e['prop_features'] = prop_features

        example_linked_list.append(e)

        print("collect {}".format(context))

    avg_n_ent = (sum([len(e['entities']) for e in example_linked_list]) * 1.0 /
                 len(example_linked_list))

    print('Average number of entities is {}'.format(avg_n_ent))
    if args.expand_entities:
        expand_entities(args, example_linked_list, table_dict)
        avg_n_ent = (sum([len(e['entities']) for e in example_linked_list]) * 1.0 /
                     len(example_linked_list))
        print('After expanding, average number of entities is {}'.format(avg_n_ent))

    for e in example_linked_list:
        e['tmp_tokens'] = e['tokens'][:]

    if args.anonymize_datetime_and_number_entities:
        for e in example_linked_list:
            for ent in e['entities']:
                if ent['type'] == 'datetime_list':
                    for t in range(ent['token_start'], ent['token_end']):
                        e['tmp_tokens'][t] = '<DECODE>'
                elif ent['type'] == 'num_list':
                    for t in range(ent['token_start'], ent['token_end']):
                        e['tmp_tokens'][t] = '<START>'

    if args.merge_entities:
      merge_entities(args, example_linked_list, table_dict)
      avg_n_ent = (sum([len(e['entities']) for e in example_linked_list]) * 1.0 /
                   len(example_linked_list))
      print('After merging, average number of entities is {}'.format(avg_n_ent))

    if args.process_conjunction:
        n_conjunction = process_conjunction(example_linked_list, 'or')
        print('{} conjunctions processed.'.format(n_conjunction))
        avg_n_ent = (sum([len(e['entities']) for e in example_linked_list]) * 1.0 /
                     len(example_linked_list))
        print('After processing conjunction, average number of entities is {}'.format(avg_n_ent))

    for e in example_linked_list:
        e['tokens'] = e['tmp_tokens']

    return example_linked_list


def dump_examples(examples, fn):
    """ Dump samples into .jsonl, which is input format of MAPO"""
    t1 = time.time()
    with open(fn, 'w') as f:
        for i, e in enumerate(examples):
            f.write(json.dumps(e))
            f.write('\n')
    t2 = time.time()
    print('{} sec used dumping {} examples.'.format(t2 - t1, len(examples)))


def merge_entities(args, examples, table_dict):
    for e in examples:
        ents = [ent for ent in e['entities']
                if ent['type'] == 'string_list' and ent['value'][0]]
        other_ents = [ent for ent in e['entities'] if ent['type'] != 'string_list']
        kg = table_dict[e['context']]
        l = len(ents)
        new_ents = []
        i = 0
        merged = False
        while i < l:
            top_ent = ents[i].copy()
            new_ents.append(top_ent)
            i += 1
            while i < l:
                if ents[i]['token_start'] - top_ent['token_end'] <= 2:
                    tokens = [tk for tk in
                              e['tokens'][top_ent['token_start']:ents[i]['token_end']]]
                    ent_tokens = [top_ent['value'][0],
                                  ents[i]['value'][0]]
                    new_str_1 = u' '.join(tokens)
                    new_str_2 = u' '.join(ent_tokens)
                    new_str_3 = u'-'.join(tokens)
                    new_str_4 = u'-'.join(ent_tokens)
                    new_str_5 = u''.join(tokens)
                    new_str_6 = u''.join(ent_tokens)
                    if string_in_table(new_str_1, kg, args.use_tokens_contain):
                        new_str = new_str_1
                    elif string_in_table(new_str_2, kg, args.use_tokens_contain):
                        new_str = new_str_2
                    elif string_in_table(new_str_3, kg, args.use_tokens_contain):
                        new_str = new_str_3
                    elif string_in_table(new_str_4, kg, args.use_tokens_contain):
                        new_str = new_str_4
                    elif string_in_table(new_str_5, kg, args.use_tokens_contain):
                        new_str = new_str_5
                    elif string_in_table(new_str_6, kg, args.use_tokens_contain):
                        new_str = new_str_6
                    else:
                        new_str = ''
                    if new_str:
                        top_ent = dict(value=[new_str], type='string_list',
                                       token_start=top_ent['token_start'],
                                       token_end=ents[i]['token_end'])
                        new_ents[-1] = top_ent
                        i += 1
                    else:
                        break
                else:
                    break
        e['entities'] = new_ents + other_ents
        for ent in e['entities']:
            for t in range(ent['token_start'], ent['token_end']):
                e['features'][t] = [1]


def expand_entities(args, examples, table_dict):
    for e in examples:
        # for e in [example_dict['nt-11874']]:
        ents = [ent for ent in e['entities']
                if ent['type'] == 'string_list' and ent['value'][0]]
        other_ents = [ent for ent in e['entities'] if ent['type'] != 'string_list']
        kg = table_dict[e['context']]
        l = len(ents)
        new_ents = []
        i = 0
        tokens = e['tokens']
        for ent in ents:
            # relies on order.
            if new_ents and ent['token_end'] <= new_ents[-1]['token_end']:
                continue
            else:
                ent['value'][0] = tokens[ent['token_start']]
                new_ents.append(ent)
                while True and ent['token_end'] < len(tokens):
                    new_str_list = (
                            [s.join([ent['value'][0],
                                     tokens[ent['token_end']]])
                             for s in [u' ', u'-', u'']] +
                            [s.join([ent['value'][0],
                                     normalize(tokens[ent['token_end']])])
                             for s in [u' ', u'-', u'']] +
                            [s.join([normalize(ent['value'][0]),
                                     tokens[ent['token_end']]])
                             for s in [u' ', u'-', u'']] +
                            [s.join([normalize(ent['value'][0]),
                                     normalize(tokens[ent['token_end']])])
                             for s in [u' ', u'-', u'']])
                    for new_str in new_str_list:
                        if string_in_table(new_str, kg, args.use_tokens_contain):
                            ent['token_end'] += 1
                            ent['value'] = [new_str]
                            break
                    else:
                        break
                ent['value'] = [normalize(ent['value'][0])]
        e['entities'] = new_ents + other_ents


def process_conjunction(examples, conjunction_word, other_words=None):
    i = 0
    for e in examples:
        str_ents = [ent for ent in e['entities'] if ent['type'] == 'string_list']
        other_ents = [ent for ent in e['entities'] if ent['type'] != 'string_list']
        if other_words is not None:
            extra_condition = any([w in e['tokens'] for w in other_words])
        else:
            extra_condition = True
        if str_ents and conjunction_word in e['tokens'] and extra_condition:
            or_idx = e['tokens'].index(conjunction_word)
            before_ent = None
            before_id = None
            after_ent = None
            after_id = None
            for k, ent in enumerate(str_ents):
                if ent['token_end'] <= or_idx:
                    before_ent = ent
                    before_id = k
                    before_distance = abs(ent['token_end'] - or_idx)
                if after_ent is None and ent['token_start'] > or_idx:
                    after_ent = ent
                    after_id = k
                    after_distance = abs(ent['token_start'] - or_idx)
            if (not before_ent is None and not after_ent is None and
                    before_distance <= 2 and after_distance <= 2):
                i += 1
                new_ent = dict(
                    value=before_ent['value'] + after_ent['value'],
                    # ['entity_a', 'entity_b']，decoder可以生成。这是用string_list而不用string的原因。
                    type='string_list',
                    token_start=before_ent['token_start'],
                    token_end=after_ent['token_end'])
                str_ents[before_id] = new_ent
                del str_ents[after_id]
                e['entities'] = str_ents + other_ents
    return i


def label_sample_with_corenlp(sample, client):
    """ Label question tokens with CoreNLP."""
    annotation = client.annotate(sample['question'])
    tokens, lemma_tokens, pos_tags, ner_tags, ner_vals = [], [], [], [], []
    for sentence in annotation.sentence:
        for token in sentence.token:
            tokens.append(token.word)
            lemma_tokens.append(token.lemma)
            pos_tags.append(token.pos)
            ner_tags.append(token.ner)
            ner_vals.append(token.normalizedNER)
    sample['tokens'] = '|'.join(tokens)
    sample['lemma_tokens'] = '|'.join(lemma_tokens)
    sample['pos_tags'] = '|'.join(pos_tags)
    sample['ner_tags'] = '|'.join(ner_tags)
    sample['ner_vals'] = '|'.join(ner_vals)


def main():
    # Paths
    data_folder = os.path.join(args.root_dir, args.data_dir, 'tables/hmt/')
    stop_words_file = os.path.join(args.root_dir, args.data_dir, 'stop_words.json')
    train_file = os.path.join(args.root_dir, args.data_dir, 'train_samples.jsonl')
    dev_file = os.path.join(args.root_dir, args.data_dir, 'dev_samples.jsonl')
    test_file = os.path.join(args.root_dir, args.data_dir, 'test_samples.jsonl')
    table_output_file = os.path.join(args.root_dir, args.data_dir, args.processed_input_dir, 'tables.jsonl')
    train_output_file = os.path.join(args.root_dir, args.data_dir, args.processed_input_dir, 'train_samples_processed.jsonl')
    dev_output_file = os.path.join(args.root_dir, args.data_dir, args.processed_input_dir, 'dev_samples_processed.jsonl')
    test_output_file = os.path.join(args.root_dir, args.data_dir, args.processed_input_dir, 'test_samples_processed.jsonl')

    # Preprocess the tables
    table_dict = {}
    folders = []
    t1 = time.time()
    for fn in os.listdir(os.path.join(data_folder)):
        full_path = os.path.join(data_folder, fn)
        table_name = fn.split('.')[0]
        folders.append(full_path)
        with open(full_path) as f:
            table_info = json.load(f)
        kg = table2kg(table_info)
        kg['name'] = table_name
        table_dict[table_name] = kg
    t2 = time.time()
    print('{} sec used processing the tables.'.format(t2 - t1))

    # Save the preprocessed tables as tables.jsonl
    t1 = time.time()
    with open(table_output_file, 'w') as f:
        for i, (k, v) in enumerate(table_dict.items()):
            if i % 1000 == 0:
                print('number {}'.format(i))
            f.write(json.dumps(v))
            f.write('\n')
    t2 = time.time()
    print('{} sec used dumping tables'.format(t2 - t1))

    train_sample_dict = create_dict_from_questions(train_file)

    with open(stop_words_file, 'r') as f:
        stop_words_list = json.load(f)
    stop_words = set(stop_words_list)

    # Preprocess and save train data in one as train_samples_linked.jsonl
    t1 = time.time()
    train_examples = collect_examples_from_dict(
        args, train_sample_dict, table_dict, stop_words)
    t2 = time.time()
    print('{} sec used collecting train examples.'.format(t2 - t1))
    dump_examples(train_examples, train_output_file)

    # Save train data into splits
    train_shards = []
    for i in range(args.n_train_shard):
        train_shards.append([])
    for i, e in enumerate(train_examples):
        train_shards[i % args.n_train_shard].append(e)
    for i, sh in enumerate(train_shards):
        train_shard_jsonl = os.path.join(
            args.root_dir, args.data_dir, args.processed_input_dir, 'no_split/train_split_shard_{}-{}.jsonl'.format(
                args.n_train_shard, i))
        dump_examples(sh, train_shard_jsonl)

    # Preprocess and save dev data as dev_samples_linked.jsonl
    dev_sample_dict = create_dict_from_questions(dev_file)
    t1 = time.time()
    dev_examples = collect_examples_from_dict(
        args, dev_sample_dict, table_dict, stop_words)
    t2 = time.time()
    print('{} sec used collecting dev examples.'.format(t2 - t1))
    dump_examples(dev_examples, dev_output_file)

    # Preprocess and save test data as test_samples_linked.jsonl
    test_sample_dict = create_dict_from_questions(test_file)
    t1 = time.time()
    test_examples = collect_examples_from_dict(
        args, test_sample_dict, table_dict, stop_words)
    t2 = time.time()
    print('{} sec used collecting test examples.'.format(t2 - t1))
    dump_examples(test_examples, test_output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/MY_PATH_TO/HiTab/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--processed_input_dir', type=str, default='processed_input/', help='processed input directory')
    parser.add_argument('--n_train_shard', type=int, default=90, help='num of shards of train splits')
    parser.add_argument('--max_n_tokens_for_num_prop', type=int, default=10)
    parser.add_argument('--min_frac_for_ordered_prop', type=float, default=0.2)
    parser.add_argument('--en_min_tk_count', type=int, default=10)
    parser.add_argument('--use_prop_match_count_feature', type=bool, default=False)
    parser.add_argument('--anonymize_in_table_tokens', type=bool, default=False)
    parser.add_argument('--anonymize_datetime_and_number_entities', type=bool, default=False)
    parser.add_argument('--merge_entities', type=bool, default=False)
    parser.add_argument('--process_conjunction', type=bool, default=False)
    parser.add_argument('--expand_entities', type=bool, default=True)
    parser.add_argument('--use_tokens_contain', type=bool, default=True)
    args = parser.parse_args()
    args.client = CoreNLPClient(
        annotators=['tokenize', 'pos', 'ner'],
        timeout=30000,
        memory='16G')
    main()

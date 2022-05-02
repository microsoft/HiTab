"""Parse a sample using stanford stanza tool."""

import stanza
# stanza.download('en')
nlp = stanza.Pipeline('en')

def parse_str_item(s, vocab_counter=None):
    doc = nlp(s.strip())
    doc_words = [ w.text for sent in doc.sentences
        for w in sent.words]
    doc_words = [dw.strip().lower() for dw in doc_words]
    doc_words = [dw for dw in doc_words if dw!='']
    if vocab_counter is not None:
        vocab_counter.update(doc_words)
    return doc_words

def parse_str_list(string_list, vocab_counter=None):
    parsed_string_list = []
    for string in string_list:
        doc_words = parse_str_item(string, vocab_counter)
        parsed_string_list.append(doc_words)
    return parsed_string_list

def parse_fielded_list(fielded_list, vocab_counter=None):
    parsed_fielded_list = []
    for attr, value in fielded_list:
        value_words = parse_str_item(value, vocab_counter)
        parsed_fielded_list.append( (attr, value_words) )
    return parsed_fielded_list


from typing import Dict 

def parse_sample_dict(sample: Dict, vocab_counter: Dict = None) -> Dict: 
    """Parse a processed sample with pointer-generator vocab. 
    args: 
        sample = Dict{
            'source': str, 
            'target': str, 
            'table_parent': List[List[str,str]] 
        } 
    rets: 
        parsed_sample = Dict{
            'source': List[str], 
            'target': List[str], 
            'table_parent': List[List[List[str], List[str]]] 
        }
    """ 
    source_words = parse_str_item(sample['source'], vocab_counter) 
    target_words = parse_str_item(sample['target'], vocab_counter) 
    parent_words = parse_fielded_list(sample['table_parent'], vocab_counter) 
    parsed_sample = {
        'source': source_words, 
        'target': target_words, 
        'table_parent': parent_words, 
    } 
    return parsed_sample 


import json

def parse_datafile(infile: str, outfile: str, vocab_counter: Dict = None, report_steps: int = 1000) -> None:
    """Parse the in-file dataset, write into the out-file, update the vocab-counter."""
    
    output_instances = []
    with open(infile, 'r', encoding='utf-8') as fr:
        for idx, line in enumerate(fr):
            inins = json.loads(line.strip())
            outins = parse_sample_dict(inins, vocab_counter)
            output_instances.append(outins) 

            if (idx + 1) % report_steps == 0: 
                print(f'successfully parsed {idx+1} samples..')
    
    with open(outfile, 'w', encoding='utf-8') as fw:
        for idx, outins in enumerate(output_instances):
            outline = json.dumps(outins)
            fw.write(outline + '\n') 

            if (idx + 1) % report_steps == 0: 
                print(f'successfully wrote {idx+1} samples..')

    print(f'Finished! from [{infile}] to [{outfile}]') 


import argparse 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--infile_list', type=str, nargs='+', required=True) 
    parser.add_argument('--outfile_list', type=str, nargs='+', required=True) 
    args = parser.parse_args() 

    if len(args.infile_list) != len(args.outfile_list): 
        print(f'unmatched {len(args.infile_list)} inputs and {len(args.outfile_list)} outputs. ')

    for infile, outfile in zip(args.infile_list, args.outfile_list): 
        parse_datafile(infile, outfile)
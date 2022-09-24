import torch
from transformers import GPT2Tokenizer, GPT2Model
from extractor.extractor import extract_relations
from extractor.document import Document
from dirmultigraph import visualize
import networkx as nx
import pickle
import argparse
import sys

def run(cfg):
    # load model
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.ckpts)
    model = GPT2Model.from_pretrained(cfg.ckpts)

    # load graph
    if cfg.graph is not None:
        assert 'p' and 'k' and 'l' in args.graph, 'convo must be a path to a pickle file'
        with open(args.graph, 'rb') as f:
            G = pickle.load(f)
        assert isinstance(G, nx.MultiDiGraph), 'object must be a MultiDiGraph'
    else:
        G = nx.MultiDiGraph()

    # visualize(G)

    step = 0
    while step < cfg.n_steps:
        # encode the new user input, add the eos_token and return a tensor in Tensorflow
        input_text = input(">> User: ")

        doc = Document(input_text)
        print(doc)
        relations = extract_relations(doc)

        print(relations)

        # parse relations
        for relation in relations:
            ln = relation.left_phrase.sentence
            rel = relation.relation_phrase.sentence
            rn = relation.right_phrase.sentence

            # add to graph
            if ln not in G:
                G.add_node(ln)
            if rn not in G:
                G.add_node(rn)
            
            G.add_edge(ln, rn, rel=rel)
        
        # for relation in relations:
        #     print(str(relation))
        # sys.exit()

        context_string = ''
        # add relations to context string
        
        new_user_input_ids = tokenizer.encode(context_string + input_text + tokenizer.eos_token)

        # append the new user input tokens to the chat history
        bot_input_ids = torch.concat([chat_history_ids, new_user_input_ids], axis=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

        step += 1

    # visualize(G)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='convo')
    parser.add_argument('--ckpts', type=str, default='microsoft/DialoGPT-small')
    parser.add_argument('--graph', type=str, default=None)
    parser.add_argument('--n_steps', type=int, default=15)
    parser.add_argument('--save_dir', type=str, default='convos/')

    args = parser.parse_args()
    run(args)

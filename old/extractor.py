import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy

import en_core_web_sm
nlp = en_core_web_sm.load()

from spacy.matcher import Matcher 
from spacy.tokens import Span 

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import List

def gen_graph_df(texts: List[str]):
    entity_pairs = [get_entities(text) for text in texts]
    relations = [get_relation(text) for text in texts]

    source = [i[0] for i in entity_pairs]
    target = [i[1] for i in entity_pairs]

    graph_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
    
    return graph_df


def gen_graph_dict(texts: List[str]):
    graph_df = gen_graph_df(texts)

    edges = {}
    nodes = {}

    for row in graph_df:
        rel = row['edge']
        pair = (row['source'], row['target'])

        if pair not in edges:
            edges[pair] = []
        
        if pair not in nodes:
            nodes[row['source']].append(pair)

        edges[pair].append(rel)

        nodes[row['source']].append(pair)
    
    return nodes, edges

def get_relation(sent):

    doc = nlp(sent)

    # Matcher class object 
    matcher = Matcher(nlp.vocab)

    #define the pattern 
    pattern = [{'DEP':'ROOT'},
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}] 

    matcher.add("matching_1", [pattern]) 

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]] 

    return span.text

def get_entities(sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence

    prefix = ""
    modifier = ""

    #############################################################

    for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

                ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            ## chunk 5  
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    #############################################################

    return [ent1.strip(), ent2.strip()]

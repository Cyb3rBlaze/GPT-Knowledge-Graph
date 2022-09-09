import matplotlib.pyplot as plt
import networkx as nx
import json

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
stop_words = set(stopwords.words('english'))


class GraphVisualization():
    def __init__(self):
        self.nodes = {}
        self.edges = {} # i can't run the program bc i don't have terminal access request for terminal access ill approve you
        self.graph = nx.Graph()
    
    def generate_graph_dict(self, sentence):
        tokenized = sent_tokenize(sentence)
        for i in tokenized:
            wordsList = nltk.word_tokenize(i)

        wordsList = [w for w in wordsList if not w in stop_words]
        outputs = nltk.pos_tag(wordsList)

        # self.edges = {('Anshul', 'dogs'): ["loves"]}
        # self.nodes = {"Anshul": [("Anshul", "dogs")], "dogs": [("Anshul", "dogs")]}

        nouns = []
        relationship = ""

        for word, label in outputs:
            if "NN" in label or "PRP" in label:
                nouns += [word]
            elif "JJ" in label or "VB" in label:
                relationship = word
        
        for i in range(len(nouns)-1):
            self.append_data([(nouns[i], nouns[i+1]), relationship])
            

    def append_data(self, edge_connection):
        # edge_connection structure: [(node1, node2), relationship]

        # update nodes
        node1 = edge_connection[0][0]
        node2 = edge_connection[0][1]

        if node1 not in self.nodes.keys():
            self.nodes[node1] = [edge_connection[0]]
        else:
            self.nodes[node1].append(edge_connection[0])
        
        if node2 not in self.nodes.keys():
            self.nodes[node2] = [edge_connection[0]]
        else:
            self.nodes[node2].append(edge_connection[0])

        # update edges
        if edge_connection[0] not in self.edges.keys():
            self.edges[edge_connection[0]] = [edge_connection[1]]
        else:
            self.edges[edge_connection[0]].append(edge_connection[1])
    
    def pull_data(self, node):
        if node in self.nodes.keys():
            relationships = self.nodes[node]
            return_sequence = []
            #outputs a list of sentences which can be directly tokenized and prepended to prompt
            for i in range(len(relationships)):
                for j in self.edges[relationships[i]]:
                    return_sequence += [relationships[i][0] + " " + j + " " + relationships[i][1]]
            return return_sequence
        return []
    
    def load_graph_from_json(self, json_path):
        knowledge_graph = {}
        with open(json_path) as f:
            knowledge_graph = json.load(f)
        
        nodes = knowledge_graph['nodes']
        edges = knowledge_graph['edges']
        for node, relationship in nodes:
            self.nodes[node] = relationship
        for relation, edge in edges:
            self.edges[relation] = edge

    def graph_to_json(self, out_path):
        graph = {
                "nodes": {},
                "edges": {}
            }
        graph["nodes"] = self.nodes
        graph["edges"] = self.edges
        json_object = json.dumps(graph)
        with open(out_path, 'w') as f:
            f.write(json_object)
            

    def draw_graph(self):
        self.graph.add_edges_from(self.edges.keys())
        pos = nx.spring_layout(self.graph)

        fig, ax = plt.subplots(figsize=(12,12))
        nx.draw(
            self.graph, pos, edge_color='#696969', width=1, linewidths=1,
            node_size=500, node_color='#696969', alpha=0.9, font_color="#b0b0b0",
            labels={node: node for node in self.graph.nodes()}
        )
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels=self.edges,
            font_color='#b0b0b0',
            bbox=dict(facecolor='#121212', edgecolor='#121212'))
        ax.set_facecolor('#121212')
        ax.axis('off')
        fig.set_facecolor('#121212')
        plt.show()


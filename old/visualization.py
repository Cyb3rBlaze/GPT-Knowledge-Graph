import matplotlib.pyplot as plt
import networkx as nx

from graph import gen_graph_dict

class GraphVisualization():
    def __init__(self):
        self.edges = {('Anshul', 'dogs'): ["loves"]}
        self.nodes = {"Anshul": [("Anshul", "dogs")], "dogs": [("Anshul", "dogs")]}
        self.graph = nx.Graph()

    def append_data(self, edge_connection):
        #edge_connection structure: [(node1, node2), relationship]
        
        #if 
        self.nodes[edge_connection[0][0]]
    
    def pull_data(self, node):
        relationships = self.nodes[node]
        return_sequence = []
        #outputs a list of sentences which can be directly tokenized and prepended to prompt
        for i in range(len(relationships)):
            for j in self.edges[relationships[i]]:
                return_sequence += [relationships[i][0] + " " + j + " " + relationships[i][1]]
        return return_sequence

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

visualization = GraphVisualization()
visualization.draw_graph()

print(visualization.pull_data("Anshul"))
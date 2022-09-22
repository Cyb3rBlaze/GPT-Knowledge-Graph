import matplotlib.pyplot as plt
import networkx as nx    

def visualize(G):
    pos = nx.spring_layout(G)

    fig, ax = plt.subplots(figsize=(12,12))
    nx.draw(
        G, pos, edge_color='#696969', width=1, linewidths=1,
        node_size=500, node_color='#696969', alpha=0.9, font_color="#b0b0b0",
        labels={node: node for node in G.nodes()}
    )
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=G.edges(data="rel", keys=True),
        font_color='#b0b0b0',
        bbox=dict(facecolor='#121212', edgecolor='#121212'))
    ax.set_facecolor('#121212')
    ax.axis('off')
    fig.set_facecolor('#121212')
    plt.show()


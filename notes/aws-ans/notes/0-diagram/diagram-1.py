
import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Define nodes and labels
nodes = {
    "London Office": "London Office\n(New DX Connection)",
    "DXGW": "Direct Connect\nGateway (Global)",
    "TGW-EU": "Transit Gateway\n(eu-west-2)",
    "TGW-US": "Transit Gateway\n(us-east-1)",
    "VPCs-EU": "VPCs in\nEU (eu-west-2)",
    "VPCs-US": "VPCs in\nUS (us-east-1)",
    "US Data Center": "US Data Center\n(Existing DX)"
}

# Add nodes to the graph
G.add_nodes_from(nodes.keys())

# Define edges to represent connections
edges = [
    ("London Office", "DXGW"),
    ("DXGW", "TGW-EU"),
    ("TGW-EU", "VPCs-EU"),
    ("TGW-EU", "TGW-US"),
    ("TGW-US", "VPCs-US"),
    ("US Data Center", "DXGW")
]

# Add edges to the graph
G.add_edges_from(edges)

# Set up plot
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  # For consistent layout

# Draw nodes, edges, and labels
nx.draw(G, pos, with_labels=True, labels=nodes, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)
nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
plt.title("AWS Multiregion DX + TGW Connectivity (Option B)", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()
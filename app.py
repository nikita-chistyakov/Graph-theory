"""
Dijkstra's Algorithm Visualization App

A Streamlit application for visualizing Dijkstra's algorithm on graphs.
This module focuses on the UI and visualization, while the algorithm 
implementation is imported from dijkstra.py.
"""

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import defaultdict

# Import the Dijkstra implementation
from dijkstra_cities import dijkstra_step_by_step, reconstruct_path, EXAMPLE_GRAPHS

# Title and description
st.title("Dijkstra's Algorithm Visualization")
st.write("""
This app demonstrates how Dijkstra's algorithm finds the shortest path between nodes in a graph.
You can customize the graph, select start and end nodes, and watch the algorithm in action.
""")

# Choose graph
graph_choice = st.selectbox(
    "Select a graph example:",
    list(EXAMPLE_GRAPHS.keys())
)

# Get the selected graph
current_graph = EXAMPLE_GRAPHS[graph_choice]

# Custom graph option
if st.checkbox("Create custom graph instead"):
    st.subheader("Custom Graph Creator")
    
    # Initialize or get existing custom graph
    if 'custom_graph' not in st.session_state:
        st.session_state.custom_graph = {'A': {'B': 1}, 'B': {'A': 1}}
    
    # Node management
    col1, col2 = st.columns(2)
    
    with col1:
        new_node = st.text_input("Add new node:", key="new_node")
        if st.button("Add Node"):
            if new_node and new_node not in st.session_state.custom_graph:
                st.session_state.custom_graph[new_node] = {}
                st.success(f"Node {new_node} added!")
            else:
                st.error("Node already exists or invalid name")
    
    with col2:
        nodes_to_remove = st.multiselect("Select nodes to remove:", 
                                         list(st.session_state.custom_graph.keys()),
                                         key="remove_nodes")
        if st.button("Remove Selected Nodes"):
            for node in nodes_to_remove:
                if node in st.session_state.custom_graph:
                    del st.session_state.custom_graph[node]
                    # Remove edges to this node
                    for src in st.session_state.custom_graph:
                        if node in st.session_state.custom_graph[src]:
                            del st.session_state.custom_graph[src][node]
            st.success("Nodes removed!")
    
    # Edge management
    st.subheader("Add/Update Edge")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source = st.selectbox("Source node:", list(st.session_state.custom_graph.keys()), key="edge_source")
    with col2:
        target = st.selectbox("Target node:", 
                             [n for n in st.session_state.custom_graph.keys() if n != source], 
                             key="edge_target")
    with col3:
        weight = st.number_input("Weight:", min_value=1, value=1, key="edge_weight")
    
    if st.button("Add/Update Edge"):
        if source and target and source != target:
            st.session_state.custom_graph[source][target] = weight
            st.success(f"Edge from {source} to {target} with weight {weight} added/updated!")
        else:
            st.error("Invalid edge")
    
    # Option to make edges bidirectional
    if st.checkbox("Make edges bidirectional"):
        if st.button("Update edges to be bidirectional"):
            for src in list(st.session_state.custom_graph.keys()):
                for dst, w in list(st.session_state.custom_graph[src].items()):
                    if dst in st.session_state.custom_graph:
                        st.session_state.custom_graph[dst][src] = w
            st.success("All edges are now bidirectional!")
    
    # Show current custom graph
    st.subheader("Current Custom Graph")
    st.json(st.session_state.custom_graph)
    
    # Use custom graph
    current_graph = st.session_state.custom_graph

# Select start and end nodes
nodes = list(current_graph.keys())
col1, col2 = st.columns(2)

with col1:
    start_node = st.selectbox("Start node:", nodes)

with col2:
    end_node = st.selectbox("End node:", [n for n in nodes if n != start_node])

# Speed control for visualization
animation_speed = st.slider("Animation Speed:", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
delay = 1.0 / animation_speed  # Convert to delay

# Function to draw the current state of the graph
def draw_graph(G, pos, distances, predecessors, current_node, visited, queue, updated_nodes, current_edge):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Edge colors
    edge_colors = {}
    for u, v in G.edges():
        if predecessors[v] == u:
            edge_colors[(u, v)] = 'green'  # Shortest path edge
        elif current_edge and (u, v) == current_edge:
            edge_colors[(u, v)] = 'orange'  # Current edge being evaluated
        else:
            edge_colors[(u, v)] = 'gray'  # Default edge color
    
    # Draw edges with weights
    for u, v, data in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2, 
                             edge_color=edge_colors.get((u, v), 'gray'), 
                             ax=ax)
        
        # Edge labels (weights)
        edge_labels = {(u, v): data['weight'] for u, v, data in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    
    # Node colors
    node_colors = []
    for node in G.nodes():
        if node == start_node:
            node_colors.append('lime')  # Start node
        elif node == end_node:
            node_colors.append('red')   # End node
        elif node == current_node:
            node_colors.append('yellow')  # Current node being processed
        elif node in visited:
            node_colors.append('lightblue')  # Visited node
        elif node in updated_nodes:
            node_colors.append('pink')  # Just updated node
        else:
            node_colors.append('white')  # Unvisited node
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors='black', node_size=700, ax=ax)
    
    # Node labels and distance labels
    nx.draw_networkx_labels(G, pos, ax=ax)
    
    # Add distance labels above nodes
    for node, (x, y) in pos.items():
        distance = distances.get(node, float('infinity'))
        distance_label = str(distance) if distance != float('infinity') else "∞"
        ax.text(x, y + 0.1, f"d={distance_label}", 
               horizontalalignment='center', fontsize=9)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, label='Start Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='End Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Current Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Visited Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', markersize=10, label='Updated Node'),
        plt.Line2D([0], [0], color='green', lw=2, label='Shortest Path Edge'),
        plt.Line2D([0], [0], color='orange', lw=2, label='Edge Being Evaluated')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.title("Dijkstra's Algorithm Visualization", fontsize=16)
    plt.tight_layout()
    return fig

# Start visualization
if st.button("Start Dijkstra's Algorithm"):
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add edges
    for src, targets in current_graph.items():
        for target, weight in targets.items():
            G.add_edge(src, target, weight=weight)
    
    # Generate positions for nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Use Streamlit's status container for progress
    status = st.empty()
    status.info("Initializing algorithm...")
    
    # Container for step details
    step_details = st.empty()
    
    # Container for the graph visualization
    fig_placeholder = st.empty()
    
    # Initialize metrics
    metrics_container = st.container()
    col1, col2, col3 = metrics_container.columns(3)
    steps_counter = col1.empty()
    nodes_visited_counter = col2.empty()
    current_distance_metric = col3.empty()
    
    # Run Dijkstra's algorithm step by step
    steps = 0
    
    # Run the algorithm and visualize each step
    for state in dijkstra_step_by_step(current_graph, start_node, end_node):
        distances, predecessors, current_node, visited, queue, updated_nodes, current_edge = state
        steps += 1
        
        # Update metrics
        steps_counter.metric("Steps", steps)
        nodes_visited_counter.metric("Nodes Visited", len(visited))
        
        # Current minimum distance if we have a current node
        if current_node:
            current_dist = distances[current_node]
            dist_text = str(current_dist) if current_dist != float('infinity') else "∞"
            current_distance_metric.metric("Current Min Distance", dist_text)
        
        # Update status
        if current_node:
            status.info(f"Step {steps}: Processing node {current_node}")
            
            # Create step details text
            details_text = f"""
            **Current node:** {current_node}
            **Distance to {current_node}:** {distances[current_node]}
            **Visited nodes:** {', '.join(visited)}
            **Priority queue:** {sorted(queue)}
            """
            
            if current_edge:
                src, dst = current_edge
                details_text += f"\n**Evaluating edge:** {src} → {dst} (weight: {current_graph[src][dst]})"
                if dst in updated_nodes:
                    details_text += f"\n**Updated {dst} distance:** {distances[dst]} (via {src})"
            
            step_details.markdown(details_text)
        
        # Draw the current state
        fig = draw_graph(G, pos, distances, predecessors, current_node, visited, queue, updated_nodes, current_edge)
        fig_placeholder.pyplot(fig)
        plt.close(fig)
        
        # Add delay for animation
        time.sleep(delay)
    
    # Show final path
    final_path = reconstruct_path(predecessors, start_node, end_node)
    
    # Final status update
    if final_path:
        status.success(f"Algorithm completed! Shortest path found: {' → '.join(final_path)}")
        st.write(f"Total distance: {distances[end_node]}")
    else:
        status.error(f"No path exists from {start_node} to {end_node}")
    
    # Create final graph highlighting the shortest path
    G_final = nx.DiGraph()
    
    # Add all edges
    for src, targets in current_graph.items():
        for target, weight in targets.items():
            G_final.add_edge(src, target, weight=weight)
    
    # Find path edges
    path_edges = []
    if len(final_path) > 1:
        path_edges = [(final_path[i], final_path[i+1]) for i in range(len(final_path)-1)]
    
    # Final visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw non-path edges
    non_path_edges = [(u, v) for u, v in G_final.edges() if (u, v) not in path_edges]
    nx.draw_networkx_edges(G_final, pos, edgelist=non_path_edges, edge_color='gray', width=1, alpha=0.5, ax=ax)
    
    # Draw path edges
    nx.draw_networkx_edges(G_final, pos, edgelist=path_edges, edge_color='green', width=3, ax=ax)
    
    # Edge labels
    edge_labels = {(u, v): data['weight'] for u, v, data in G_final.edges(data=True)}
    nx.draw_networkx_edge_labels(G_final, pos, edge_labels=edge_labels, ax=ax)
    
    # Node colors for final graph
    node_colors = []
    for node in G_final.nodes():
        if node == start_node:
            node_colors.append('lime')
        elif node == end_node:
            node_colors.append('red')
        elif node in final_path:
            node_colors.append('lightyellow')
        else:
            node_colors.append('lightgray')
    
    # Draw nodes
    nx.draw_networkx_nodes(G_final, pos, node_color=node_colors, edgecolors='black', node_size=700, ax=ax)
    nx.draw_networkx_labels(G_final, pos, ax=ax)
    
    # Add distance labels
    for node, (x, y) in pos.items():
        distance = distances.get(node, float('infinity'))
        distance_label = str(distance) if distance != float('infinity') else "∞"
        ax.text(x, y + 0.1, f"d={distance_label}", 
               horizontalalignment='center', fontsize=9)
    
    # Legend for final graph
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, label='Start Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='End Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightyellow', markersize=10, label='Path Node'),
        plt.Line2D([0], [0], color='green', lw=2, label='Shortest Path')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.title("Final Shortest Path", fontsize=16)
    plt.tight_layout()
    
    st.pyplot(fig)
    plt.close(fig)
    
    # Show final distances table
    st.subheader("Final Distances from Start Node")
    distance_data = []
    for node, distance in distances.items():
        distance_display = str(distance) if distance != float('infinity') else "∞"
        in_path = "Yes" if node in final_path else "No"
        distance_data.append({"Node": node, "Distance": distance_display, "On Shortest Path": in_path})
    
    st.table(distance_data)

# Add information section
with st.expander("About Dijkstra's Algorithm"):
    st.markdown("""
    ### Dijkstra's Algorithm
    Dijkstra's algorithm is a graph search algorithm that solves the single-source shortest path problem for a graph with non-negative edge weights.
    
    ### How It Works
    1. Initialize distances of all vertices as infinite and distance of source vertex as 0
    2. Create a priority queue and enqueue the source vertex
    3. While the priority queue is not empty:
       a. Dequeue a vertex with the minimum distance value
       b. For each adjacent vertex, if the distance through the current vertex is shorter than its current distance, update its distance
       c. Enqueue the adjacent vertex with its updated distance
    
    ### Time Complexity
    - O((V + E) log V) with a binary heap implementation
    - O(V²) with an array implementation
    
    Where V is the number of vertices and E is the number of edges in the graph.
    
    ### Applications
    - Finding shortest paths in transportation networks
    - IP routing to find Open Shortest Path First
    - Flight scheduling
    - Robot navigation
    """)

# Footer
st.markdown("---")
st.markdown("Created with Streamlit for Dijkstra's Algorithm Visualization")

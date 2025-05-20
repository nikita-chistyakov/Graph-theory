"""
Implementation of Dijkstra's algorithm for finding shortest paths in graphs.
This module provides both standard and step-by-step implementations for visualization.
"""

import heapq
from collections import defaultdict

# Example graphs for demonstration
EXAMPLE_GRAPHS = {
    "Simple Graph": {
        'A': {'B': 4, 'C': 2},
        'B': {'A': 4, 'D': 2, 'E': 3},
        'C': {'A': 2, 'D': 4, 'F': 5},
        'D': {'B': 2, 'C': 4, 'E': 1, 'F': 1},
        'E': {'B': 3, 'D': 1, 'G': 2},
        'F': {'C': 5, 'D': 1, 'G': 1},
        'G': {'E': 2, 'F': 1}
    },
    "Road Network": {
        'Home': {'Store': 5, 'School': 7, 'Park': 3},
        'Store': {'Home': 5, 'Office': 8, 'Mall': 4},
        'School': {'Home': 7, 'Park': 2, 'Library': 3},
        'Park': {'Home': 3, 'School': 2, 'Lake': 6},
        'Office': {'Store': 8, 'Mall': 2, 'Hospital': 5},
        'Mall': {'Store': 4, 'Office': 2, 'Hospital': 3, 'Airport': 10},
        'Library': {'School': 3, 'Lake': 4, 'Hospital': 7},
        'Lake': {'Park': 6, 'Library': 4},
        'Hospital': {'Office': 5, 'Mall': 3, 'Library': 7, 'Airport': 6},
        'Airport': {'Mall': 10, 'Hospital': 6}
    },
    "Computer Network": {
        'Router1': {'Router2': 2, 'Router3': 4, 'Switch1': 1},
        'Router2': {'Router1': 2, 'Router4': 3, 'Switch2': 2},
        'Router3': {'Router1': 4, 'Router4': 1, 'Switch3': 3},
        'Router4': {'Router2': 3, 'Router3': 1, 'Switch4': 1},
        'Switch1': {'Router1': 1, 'Server1': 2, 'Server2': 3},
        'Switch2': {'Router2': 2, 'Server3': 1, 'Server4': 2},
        'Switch3': {'Router3': 3, 'Server5': 2, 'Server6': 1},
        'Switch4': {'Router4': 1, 'Server7': 3, 'Server8': 2},
        'Server1': {'Switch1': 2},
        'Server2': {'Switch1': 3},
        'Server3': {'Switch2': 1},
        'Server4': {'Switch2': 2},
        'Server5': {'Switch3': 2},
        'Server6': {'Switch3': 1},
        'Server7': {'Switch4': 3},
        'Server8': {'Switch4': 2}
    }
}

def dijkstra(graph, start, end=None):
    """
    Standard implementation of Dijkstra's algorithm.
    
    Args:
        graph: A dictionary where keys are nodes and values are dictionaries of connected nodes and weights
        start: Starting node
        end: End node (optional, if None, computes distances to all nodes)
        
    Returns:
        distances: Dictionary of shortest distances from start to each node
        predecessors: Dictionary of predecessors for each node
    """
    # Initialize
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    predecessors = {node: None for node in graph}
    priority_queue = [(0, start)]
    visited = set()
    
    # Main loop
    while priority_queue:
        # Get node with smallest distance
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # Skip if we've already processed this node or found the end
        if current_node in visited:
            continue
        if current_node == end:
            break
            
        visited.add(current_node)
        
        # Check neighbors
        for neighbor, weight in graph[current_node].items():
            if neighbor in visited:
                continue
                
            distance = current_distance + weight
            
            # If we found a shorter path, update
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances, predecessors

def reconstruct_path(predecessors, start, end):
    """
    Reconstruct the path from start to end using the predecessors dictionary.
    
    Args:
        predecessors: Dictionary of node predecessors
        start: Start node
        end: End node
        
    Returns:
        list: Path from start to end, or empty list if no path exists
    """
    if predecessors[end] is None and end != start:
        return []  # No path exists
        
    path = [end]
    while path[-1] != start:
        path.append(predecessors[path[-1]])
    
    return list(reversed(path))

def dijkstra_step_by_step(graph, start, end=None):
    """
    Step-by-step implementation of Dijkstra's algorithm that yields state after each operation.
    
    Args:
        graph: A dictionary where keys are nodes and values are dictionaries of connected nodes and weights
        start: Starting node
        end: End node (optional, if None, computes distances to all nodes)
        
    Yields:
        tuple: (
            distances: Current distances dictionary,
            predecessors: Current predecessors dictionary,
            current_node: Node currently being processed,
            visited: Set of visited nodes,
            queue: Current priority queue,
            updated_nodes: Set of nodes updated in current step,
            current_edge: The edge being evaluated in the current step
        )
    """
    # Initialize
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    predecessors = {node: None for node in graph}
    priority_queue = [(0, start)]
    visited = set()
    
    # Initial state - nothing processed yet
    yield distances, predecessors, None, visited, [(0, start)], set(), None
    
    # Main loop
    while priority_queue:
        # Get node with smallest distance
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # Skip if we've already processed this node
        if current_node in visited:
            continue
            
        # Yield state with current node selected, no updates yet
        yield distances, predecessors, current_node, visited, priority_queue, set(), None
        
        # Mark node as visited
        visited.add(current_node)
        
        # Early termination if we've reached the end
        if current_node == end:
            break
        
        # Process neighbors
        for neighbor, weight in graph[current_node].items():
            # Skip if neighbor already processed
            if neighbor in visited:
                continue
                
            # Show edge being evaluated
            current_edge = (current_node, neighbor)
            yield distances, predecessors, current_node, visited, priority_queue, set(), current_edge
            
            # Calculate potential new distance
            distance = current_distance + weight
            
            # Check if we found a shorter path
            updated_nodes = set()
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
                updated_nodes.add(neighbor)
                
                # Show state after update
                yield distances, predecessors, current_node, visited, priority_queue, updated_nodes, current_edge
    
    # Final state
    yield distances, predecessors, None, visited, priority_queue, set(), None

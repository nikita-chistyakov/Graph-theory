"""
Dijkstra's Algorithm Implementation Module

This module contains implementations of Dijkstra's algorithm for finding shortest paths in graphs,
including a step-by-step version for visualization.
"""

import heapq
from typing import Dict, List, Tuple, Set, Optional, Any

def dijkstra(graph: Dict[str, Dict[str, int]], start: str, end: str = None) -> Tuple[Dict[str, int], Dict[str, Optional[str]]]:
    """
    Standard implementation of Dijkstra's algorithm for finding shortest paths.
    
    Args:
        graph: A dictionary where keys are node names and values are dictionaries
               mapping neighboring nodes to edge weights.
        start: The starting node name.
        end: Optional end node for early termination.
        
    Returns:
        A tuple containing:
        - A dictionary mapping each node to its shortest distance from the start node
        - A dictionary mapping each node to its predecessor in the shortest path
    """
    # Initialize distances with infinity for all nodes except the start node
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    
    # Initialize predecessors dictionary to track the path
    predecessors = {node: None for node in graph}
    
    # Priority queue to store vertices that need to be processed
    # Format: (distance, node)
    priority_queue = [(0, start)]
    
    # Set to keep track of visited nodes
    visited = set()
    
    while priority_queue:
        # Get the node with the smallest distance
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # If we've already processed this node, skip it
        if current_node in visited:
            continue
            
        # Mark the current node as visited
        visited.add(current_node)
        
        # If we've reached the destination, we can stop
        if end and current_node == end:
            break
        
        # If the current distance is greater than the known distance, skip
        if current_distance > distances[current_node]:
            continue
            
        # Check all neighbors of the current node
        for neighbor, weight in graph[current_node].items():
            # Calculate the distance to the neighbor through the current node
            distance = current_distance + weight
            
            # If we found a shorter path to the neighbor, update it
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
                
    return distances, predecessors

def dijkstra_step_by_step(graph: Dict[str, Dict[str, int]], start: str, end: str = None):
    """
    Step-by-step implementation of Dijkstra's algorithm for visualization.
    
    Args:
        graph: A dictionary where keys are node names and values are dictionaries
               mapping neighboring nodes to edge weights.
        start: The starting node name.
        end: Optional end node for early termination.
        
    Yields:
        A tuple containing:
        - Current distances dictionary
        - Current predecessors dictionary
        - Current node being processed
        - Set of visited nodes
        - Current priority queue
        - Set of nodes that were just updated
        - Current edge being evaluated (or None)
    """
    # Initialize distances with infinity for all nodes except the start node
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    
    # Initialize predecessors dictionary to track the path
    predecessors = {node: None for node in graph}
    
    # Priority queue to store vertices that need to be processed
    # Format: (distance, node)
    priority_queue = [(0, start)]
    
    # Set to keep track of visited nodes
    visited = set()
    
    # Initial state
    yield (distances.copy(), predecessors.copy(), None, visited.copy(), 
           [(0, start)], set(), None)
    
    while priority_queue:
        # Get the node with the smallest distance
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # Skip if already visited
        if current_node in visited:
            continue
            
        # Check if we've reached the destination
        if end and current_node == end:
            yield (distances.copy(), predecessors.copy(), current_node, 
                   visited.copy(), [(d, n) for d, n in priority_queue], 
                   set(), None)
            break
            
        # Mark the current node as visited
        visited.add(current_node)
        
        # Skip if current distance is greater than known distance
        if current_distance > distances[current_node]:
            continue
            
        # Yield current state before processing neighbors
        yield (distances.copy(), predecessors.copy(), current_node, 
               visited.copy(), [(d, n) for d, n in priority_queue], 
               set(), None)
        
        # Check all neighbors
        for neighbor, weight in graph[current_node].items():
            if neighbor in visited:
                continue
                
            # Calculate new distance
            distance = current_distance + weight
            
            # Current edge being considered
            current_edge = (current_node, neighbor)
            
            # If we found a shorter path to the neighbor, update it
            if distance < distances[neighbor]:
                old_distance = distances[neighbor]
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
                
                # Yield state after updating a neighbor
                yield (distances.copy(), predecessors.copy(), current_node, 
                       visited.copy(), [(d, n) for d, n in priority_queue], 
                       {neighbor}, current_edge)

def reconstruct_path(predecessors: Dict[str, Optional[str]], start: str, end: str) -> List[str]:
    """
    Reconstruct the shortest path from start to end using the predecessors dictionary.
    
    Args:
        predecessors: Dictionary mapping each node to its predecessor
        start: Starting node
        end: Ending node
        
    Returns:
        A list representing the shortest path from start to end
    """
    path = []
    current = end
    
    # If end is not reachable from start
    if predecessors[end] is None and end != start:
        return []
        
    # Build the path by working backwards from the end
    while current is not None:
        path.append(current)
        current = predecessors[current]
        if current == start:  # Make sure we include the start node
            path.append(current)
            break
        
    # Reverse the path to get the correct order (start to end)
    return path[::-1]

# Sample graph data structure
EXAMPLE_GRAPHS = {
    "Simple Graph": {
        'A': {'B': 4, 'C': 2},
        'B': {'A': 4, 'C': 1, 'D': 5},
        'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
        'D': {'B': 5, 'C': 8, 'E': 2, 'F': 6},
        'E': {'C': 10, 'D': 2, 'F': 3},
        'F': {'D': 6, 'E': 3},
    },
    "City Grid": {
        'New York': {'Boston': 4, 'Washington': 5},
        'Boston': {'New York': 4, 'Chicago': 9, 'Toronto': 8},
        'Washington': {'New York': 5, 'Atlanta': 6, 'Dallas': 12},
        'Chicago': {'Boston': 9, 'Seattle': 15, 'Denver': 8},
        'Toronto': {'Boston': 8, 'Seattle': 18},
        'Atlanta': {'Washington': 6, 'Dallas': 7, 'Miami': 5},
        'Dallas': {'Washington': 12, 'Atlanta': 7, 'Denver': 9, 'Los Angeles': 12},
        'Miami': {'Atlanta': 5},
        'Denver': {'Chicago': 8, 'Dallas': 9, 'Seattle': 10, 'Los Angeles': 10, 'San Francisco': 12},
        'Seattle': {'Chicago': 15, 'Toronto': 18, 'Denver': 10, 'San Francisco': 8},
        'Los Angeles': {'Dallas': 12, 'Denver': 10, 'San Francisco': 6},
        'San Francisco': {'Seattle': 8, 'Denver': 12, 'Los Angeles': 6}
    }
}

# Example usage
if __name__ == "__main__":
    graph = EXAMPLE_GRAPHS["Simple Graph"]
    start_node = 'A'
    end_node = 'F'
    
    # Run standard Dijkstra's algorithm
    distances, predecessors = dijkstra(graph, start_node, end_node)
    
    # Reconstruct the path
    path = reconstruct_path(predecessors, start_node, end_node)
    
    # Print results
    print(f"Shortest path from {start_node} to {end_node}:")
    print(" -> ".join(path))
    print(f"Total distance: {distances[end_node]}")
    
    # Print all distances
    print("\nDistances from", start_node)
    for node, dist in distances.items():
        print(f"  to {node}: {dist}")

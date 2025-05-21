# Dijkstra's Algorithm Visualization

An interactive Streamlit application that visualizes Dijkstra's shortest path algorithm on customizable graphs.

<img width="700" alt="Screen Shot 2025-05-21 at 1 09 40 PM" src="https://github.com/user-attachments/assets/c7e25bd6-8f41-4913-aa85-95919b419cda" />

**The edge weights in the graph represent real distances between cities in miles.*

## Overview

This application demonstrates how Dijkstra's algorithm finds the shortest path between nodes in a graph. Users can:
- Select from pre-defined example graphs or create custom ones
- Choose start and end nodes
- Watch the algorithm run step-by-step with visual feedback
- Adjust animation speed
- View detailed information about each step of the process

## Features

- **Interactive Graph Creation**: Add/remove nodes and edges with custom weights
- **Real-time Visualization**: See the algorithm in action with color-coded nodes and edges
- **Step-by-step Explanation**: Follow detailed updates for each step of the algorithm
- **Multiple Example Graphs**: Choose from pre-defined graph examples
- **Dynamic Metrics**: Track algorithm progress with step counter, visited nodes, and current distances

## Files

- `app.py`: Main Streamlit application with UI components
- `dijkstra.py`: Implementation of Dijkstra's algorithm with step-by-step visualization
- `requirements.txt`: Required dependencies

## Usage

1. Select a graph from the dropdown menu called "City Grid" or create your own custom graph
2. Choose start and end cities
3. Adjust the animation speed if desired
4. Click "Start Dijkstra's Algorithm" to begin the visualization
5. Watch as the algorithm finds the shortest path (in miles) between your selected cities
6. View the final results including the shortest path and total distance in miles


## Learn More

Expand the "About Dijkstra's Algorithm" section in the app to learn about:
- The algorithm's functionality and steps
- Time complexity analysis
- Real-world applications

## Try it

- https://dijkstra.streamlit.app/

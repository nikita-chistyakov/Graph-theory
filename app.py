import streamlit as st
from dijkstra import dijkstra, reconstruct_path, EXAMPLE_GRAPHS

st.title("Dijkstra's Algorithm Visualizer")

graph = EXAMPLE_GRAPHS["Simple Graph"]
start_node = st.selectbox("Start Node", list(graph.keys()))
end_node = st.selectbox("End Node", list(graph.keys()))

if st.button("Find Shortest Path"):
    distances, predecessors = dijkstra(graph, start_node, end_node)
    path = reconstruct_path(predecessors, start_node, end_node)

    st.subheader("Shortest Path")
    if path:
        st.write(" → ".join(path))
        st.write(f"Total Distance: {distances[end_node]}")
    else:
        st.write("No path found.")

    st.subheader("All Distances")
    for node, dist in distances.items():
        st.write(f"{start_node} → {node}: {dist}")

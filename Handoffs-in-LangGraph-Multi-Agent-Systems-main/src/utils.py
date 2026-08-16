"""Utility functions for graph visualization and other helpers."""

from pathlib import Path


def save_mermaid_diagram(graph, output_path: str = "artifacts/graph_setup.png"):
    """Save the workflow graph visualization to a file.
    
    Args:
        graph: The compiled LangGraph to visualize
        output_path: Path to save the visualization PNG
    """
    try:
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Generate visualization using LangGraph's built-in method
        graph_png = graph.get_graph(xray=True).draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(graph_png)
            # print(f"Graph diagram saved as PNG: {output_path}")
    except Exception as e:
        print(f"Could not generate diagram: {e}")


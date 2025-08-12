"""Visualize the graph using Mermaid and display it as an image."""

from PIL import Image
import io

from agent.graph import graph

image_bytes = graph.get_graph().draw_mermaid_png()

image = Image.open(io.BytesIO(image_bytes))

image.show()

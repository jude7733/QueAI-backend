"""Visualize the graph using Mermaid and display it as an image."""

from PIL import Image
import io

from graph_builder import queai_graph

queai_graph.get_graph().print_ascii()

image_bytes = queai_graph.get_graph(xray=True).draw_mermaid_png()
image = Image.open(io.BytesIO(image_bytes))
image.show()

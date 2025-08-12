from typing import Annotated
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from tools import tools
from agent.chatbot import chatbot


class State(TypedDict):
    messages: Annotated[list, add_messages]
    model_name: str


graph_builder = StateGraph(State)

tool_node = ToolNode(tools)

# Nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# Edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {"tools": "tools", "__end__": END},
)
graph_builder.add_edge("tools", "chatbot")


graph = graph_builder.compile()


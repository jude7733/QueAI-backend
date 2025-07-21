from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
import dotenv

from agent.chatbot import chatbot

dotenv.load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    model_name: str


graph_builder = StateGraph(State)


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

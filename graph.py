from typing import Annotated, Optional, TypedDict
from uuid import uuid4
from langchain_core.messages import (
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    HumanMessageChunk,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph, add_messages
from agents.supervisor_agent import supervisor_agent
from agents.code_agent import code_agent
from agents.research_agent import research_agent
from agents.image_agent import image_agent
from utils import pretty_print_messages
import pprint
import sqlite3
import dotenv

dotenv.load_dotenv()

sqlite_connection = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_connection)


class GeneratedImage(TypedDict, total=False):
    data: str
    mime_type: str
    prompt: str
    file_path: Optional[str]
    url: Optional[str]
    filename: Optional[str]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    generated_image: Optional[GeneratedImage]


supervisor = (
    StateGraph(AgentState)
    .add_node(
        supervisor_agent(),
        destinations=("research_agent", "code_agent", "image_agent", END),
    )
    .add_node(research_agent())
    .add_node(code_agent())
    .add_node(image_agent)
    .add_edge(START, "supervisor")
    .add_edge("research_agent", "supervisor")
    .add_edge("code_agent", "supervisor")
    .add_edge("image_agent", "supervisor")
    .compile(checkpointer=memory)
)


def serialise_ai_message_chunk(chunk):
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    if isinstance(chunk, HumanMessageChunk):
        return chunk.content

    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialisation"
        )


if __name__ == "__main__":
    while True:
        input_query = input("Enter prompt: ")
        config = {"configurable": {"thread_id": str(uuid4())}}

        for chunk in supervisor.stream(
            {"messages": [HumanMessage(content=input_query)], "generated_image": None},
            config=config,
            stream_mode="updates",
        ):
            node_name = list(chunk)[0]
            # pprint.pprint(chunk, indent=4)
            for msg in chunk[node_name]["messages"]:
                print(25 * "+")
                print("node name: ", node_name)
                print(msg)
                print(25 * "-")

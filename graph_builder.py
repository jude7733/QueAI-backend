from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

import pprint
from typing import Annotated, Union
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from custom_tools import queai_tools
from utils import load_system_prompt
from langchain_core.agents import AgentAction, AgentFinish
import operator
import sqlite3


class State(TypedDict):
    messages: Annotated[list, add_messages]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


graph_builder = StateGraph(AgentState)

llm = init_chat_model("gemini-2.5-pro", model_provider="google_genai")
llm_with_tools = llm.bind_tools(queai_tools)


def chatbot(state: AgentState):
    system_prompt = load_system_prompt()
    messages = state.get("messages", [])

    if not any(
        isinstance(msg, SystemMessage)
        or (isinstance(msg, dict) and msg.get("role") == "system")
        for msg in messages
    ):
        messages.insert(0, SystemMessage(content=system_prompt))

    message = llm_with_tools.invoke(state["messages"])
    return {"messages": [message]}


tool_node = ToolNode(queai_tools)

# Nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# Edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")


sqlite_connection = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_connection)
queai_graph = graph_builder.compile(checkpointer=memory, debug=True)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": 1}}

    pprint.pprint(
        queai_graph.invoke(
            {"messages": HumanMessage(content="When is the next world cup?")},
            config=config,
        )
    )

    # queai_graph.get_state(config=config)

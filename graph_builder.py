import pprint

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage

from typing import Annotated
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from tools import tools


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
llm_with_tools = llm.bind_tools(tools)


def load_system_prompt(filepath="system_prompt.txt"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are a helpful assistant."


def chatbot(state: State):
    system_prompt = load_system_prompt()

    # # Get or initialize messages
    # messages = state.get("messages", [])
    #
    # # Add system prompt if not already included
    # if not any(
    #     isinstance(msg, SystemMessage)
    #     or (isinstance(msg, dict) and msg.get("role") == "system")
    #     for msg in messages
    # ):
    #     messages.insert(0, SystemMessage(content=system_prompt))

    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


tool_node = ToolNode(tools)

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


memory = InMemorySaver()
queai_graph = graph_builder.compile(checkpointer=memory)

if __name__ == "__main__":
    message = {"role": "user", "content": "Hello, who won the world series in 2020?"}
    pprint.pprint(queai_graph.invoke({"messages": [message]}))

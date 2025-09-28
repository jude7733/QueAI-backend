from typing import Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_experimental.tools import PythonREPLTool
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command


def coder_node(state: MessagesState) -> Command[Literal["validator"]]:
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    python_repl_tool = PythonREPLTool()
    code_agent = create_react_agent(
        llm,
        tools=[python_repl_tool],
        prompt=(
            "You are a coder and analyst. Focus on mathematical calculations, analyzing, solving math questions, "
            "and executing code. Handle technical problem-solving and data tasks."
        ),
    )

    result = code_agent.invoke(state)

    print("--- Workflow Transition: Coder → Validator ---")

    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="coder")
            ]
        },
        goto="validator",
    )

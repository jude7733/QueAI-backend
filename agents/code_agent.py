from langchain_groq import ChatGroq
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt import create_react_agent

python_repl_tool = PythonREPLTool()


def code_agent():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    code_agent = create_react_agent(
        llm,
        tools=[python_repl_tool],
        prompt=(
            "You are a coder and analyst. Focus on mathematical calculations, analyzing, solving math questions, "
            "and executing code. Handle technical problem-solving and data tasks."
        ),
        name="code_agent",
    )

    return code_agent

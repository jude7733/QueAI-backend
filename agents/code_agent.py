from langchain.chat_models import init_chat_model
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt import create_react_agent

python_repl_tool = PythonREPLTool()


def code_agent():
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
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

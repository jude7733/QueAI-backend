from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from handoff_tool import (
    assign_to_research_agent,
    assign_to_math_agent,
    assign_to_coder_agent,
)


def supervisor_agent():
    llm = init_chat_model("gemini-2.5-pro", model_provider="google_genai")
    supervisor_agent = create_react_agent(
        llm,
        tools=[
            assign_to_research_agent,
            assign_to_math_agent,
            assign_to_coder_agent,
        ],
        prompt=(
            "You are QueAI made by QueAI team, a supervisor managing three agents:\n"
            "- a research agent. Assign to fetch live data from internet and research-related tasks to this agent\n"
            "- a math agent. Assign math-related tasks to this agent\n"
            "- a code agent. Assign code-related execution tasks to this agent\n"
            "Assign work to one agent at a time, do not call agents in parallel.\n"
            "Do not do any work yourself."
        ),
        name="supervisor",
    )

    return supervisor_agent

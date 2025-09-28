from typing import Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from custom_tools import queai_tools


def research_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Research agent node that gathers information using Tavily search.
    Takes the current task state, performs relevant research,
    and returns findings for validation.
    """

    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    research_agent = create_react_agent(
        llm,
        tools=queai_tools,
        prompt="""You are an Information Specialist with expertise in comprehensive research. Your responsibilities include:

1. Identifying key information needs based on the query context
2. Gathering relevant, accurate, and up-to-date information from reliable sources
3. Organizing findings in a structured, easily digestible format
4. Citing sources when possible to establish credibility
5. Focusing exclusively on information gathering - avoid analysis or implementation

Provide thorough, factual responses without speculation where information is unavailable.""",
    )

    result = research_agent.invoke(state)

    print("--- Workflow Transition: Researcher → Validator ---")

    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="validator",
    )

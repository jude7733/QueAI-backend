from typing import Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.types import Command


def enhancer_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    Enhancer agent node that improves and clarifies user queries.
    Takes the original user input and transforms it into a more precise,
    actionable request before passing it to the supervisor.
    """

    system_prompt = (
        "You are a Query Refinement Specialist with expertise in transforming vague requests into precise instructions. Your responsibilities include:\n\n"
        "1. Analyzing the original query to identify key intent and requirements\n"
        "2. Resolving any ambiguities without requesting additional user input\n"
        "3. Expanding underdeveloped aspects of the query with reasonable assumptions\n"
        "4. Restructuring the query for clarity and actionability\n"
        "5. Ensuring all technical terminology is properly defined in context\n\n"
        "6. If the question isn't clear, make the best possible assumptions to fix it.\n"
        "Important: Never ask questions back to the user. Instead, make informed assumptions and create the most comprehensive version of their request possible."
    )

    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]

    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    enhanced_query = llm.invoke(messages)

    print("--- Workflow Transition: Prompt Enhancer → Supervisor ---")

    return Command(
        update={
            "messages": [HumanMessage(content=enhanced_query.content, name="enhancer")]
        },
        goto="supervisor",
    )

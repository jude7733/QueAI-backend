from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from custom_tools import queai_tools


def research_agent():
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    research_agent = create_react_agent(
        llm,
        tools=queai_tools,
        prompt=(
            "You are a research agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with research-related tasks,\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="research_agent",
    )

    return research_agent

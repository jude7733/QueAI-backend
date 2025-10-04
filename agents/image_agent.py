from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from custom_tools import image_tools


def image_agent():
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    image_agent = create_react_agent(
        llm,
        tools=image_tools,
        prompt=(
            "You are a image generating agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with image generation tasks,\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="image_agent",
    )

    return image_agent

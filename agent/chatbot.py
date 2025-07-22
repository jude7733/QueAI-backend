from langchain.chat_models import init_chat_model
from tools import tools
from langchain_core.messages import SystemMessage


def load_system_prompt(filepath="system_prompt.txt"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are a helpful assistant."


def chatbot(state):
    model_name = state.get("model_name", "gemini-2.5-flash")
    system_prompt = load_system_prompt()

    # Initialize the chat model
    llm = init_chat_model(model_name, model_provider="google_genai")
    llm_with_tools = llm.bind_tools(tools)

    # Get or initialize messages
    messages = state.get("messages", [])

    # Add system prompt if not already included
    if not any(
        isinstance(msg, SystemMessage)
        or (isinstance(msg, dict) and msg.get("role") == "system")
        for msg in messages
    ):
        messages.insert(0, SystemMessage(content=system_prompt))

    # Run model with tools
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

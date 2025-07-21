from langchain.chat_models import init_chat_model

# TODO: Add system prompt


def chatbot(state):
    model_name = state.get("model_name", "gemini-2.5-flash")
    llm = init_chat_model(f"google_genai:{model_name}")
    return {"messages": [llm.invoke(state["messages"])]}

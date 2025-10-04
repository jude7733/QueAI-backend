from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent


def add(a: float, b: float):
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float):
    """Subtract two numbers."""
    return a - b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b


def math_agent():
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

    math_agent = create_react_agent(
        llm,
        tools=[add, multiply, divide],
        prompt=(
            "You are a math agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with math-related tasks\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="math_agent",
    )

    return math_agent

from pydantic import BaseModel


class ChatRequest(BaseModel):
    user_input: str
    model_name: str = "gemini-2.5-flash"

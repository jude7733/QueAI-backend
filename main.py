from fastapi import FastAPI
from pydantic import BaseModel

# Import your LangGraph agent from another file
# from your_rag_logic import get_rag_chain

app = FastAPI()

# This is where you would load your model or RAG chain.
# Doing this at startup avoids reloading it on every request.
# rag_chain = get_rag_chain()


# Define the input data model for your API
class Query(BaseModel):
    model: str = "gemini-2.5-flash"
    question: str
    history: list = []
    user_id: str = "1"


# Define your API endpoint
@app.post("/ask")
async def run_agent(query: Query):
    """
    Receives a question and returns the RAG agent's response.
    """
    question = query.question

    # response = rag_chain.invoke({"question": question})

    # Sample
    response = (
        f"This is the answer from the LangGraph agent for the question: '{question}'"
    )

    return {"answer": response}


@app.get("/")
def read_root():
    return {"status": "RAG API is running"}

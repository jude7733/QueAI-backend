from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from agent.graph import graph
from models import ChatRequest
import dotenv

dotenv.load_dotenv()


app = FastAPI()


# Send History along with user input
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    events = graph.stream(
        {
            "messages": [{"role": "user", "content": request.user_input}],
            "model_name": request.model_name,
        }
    )
    responses = [
        value["messages"][-1].content for event in events for value in event.values()
    ]
    return {"responses": responses}


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    def event_generator():
        for event in graph.stream(
            {
                "messages": [{"role": "user", "content": request.user_input}],
                "model_name": request.model_name,
            }
        ):
            for value in event.values():
                yield value["messages"][-1].content + "\n"

    return StreamingResponse(event_generator(), media_type="text/plain")


@app.get("/")
def read_root():
    return {"status": "RAG API is running"}

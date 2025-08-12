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
    final_state = None
    for event in graph.stream(
        {
            "messages": [{"role": "user", "content": request.user_input}],
            "model_name": request.model_name,
        }
    ):
        final_state = event

    last_message = final_state[list(final_state.keys())[0]]["messages"][-1]
    return {"responses": [last_message.content]}


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    def event_generator():
        for event in graph.stream(
            {
                "messages": [{"role": "user", "content": request.user_input}],
                "model_name": request.model_name,
            }
        ):
            if "chatbot" in event:
                yield event["chatbot"]["messages"][-1].content

    return StreamingResponse(event_generator(), media_type="text/plain")


@app.get("/")
def read_root():
    return {"status": "RAG API is running"}

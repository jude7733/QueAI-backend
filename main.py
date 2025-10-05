from uuid import uuid4
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from graph import supervisor
import dotenv
import json
from pydantic import BaseModel


dotenv.load_dotenv()


class ChatRequest(BaseModel):
    user_input: str
    thread_id: str | None = uuid4().hex


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)


async def event_generator(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    for chunk in supervisor.stream(
        {
            "messages": [HumanMessage(content=request.user_input)],
        },
        config=config,
        stream_mode="messages",
    ):
        if not (isinstance(chunk, tuple) and len(chunk) == 2):
            continue

        message, metadata = chunk

        event_data = {
            "node_name": metadata.get("langgraph_node", "unknown"),
            "message_type": message.type,
            "thread_id": metadata.get("thread_id"),
            "payload": {},
        }

        content = getattr(message, "content", None)
        if content:
            event_data["payload"]["content"] = content

        tool_calls = getattr(message, "tool_calls", [])
        if tool_calls:
            event_data["payload"]["tool_calls"] = tool_calls

        if isinstance(message, ToolMessage):
            event_data["payload"]["tool_call_id"] = message.tool_call_id

        if not event_data["payload"]:
            continue

        yield json.dumps(event_data, indent=2)


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    return StreamingResponse(
        event_generator(request=request),
        media_type="text/event-stream",
    )


@app.get("/")
def read_root():
    return {"status": "RAG API is running, View at queai.vercel.app"}

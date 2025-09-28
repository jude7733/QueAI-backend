import pprint
from uuid import uuid4
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from graph_builder import queai_graph
import dotenv
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
    for chunk in queai_graph.stream(
        {
            "messages": [HumanMessage(content=request.user_input)],
        },
        config=config,
        stream_mode="custom",
    ):
        pprint.pprint(chunk)
        yield str(chunk)


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    return StreamingResponse(event_generator(request=request), media_type="text/plain")


@app.get("/")
def read_root():
    return {"status": "RAG API is running, View at queai.vercel.app"}

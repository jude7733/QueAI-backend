from uuid import uuid4
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from graph import supervisor
import pprint
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


def messages_to_json(messages):
    """Convert a list of message objects to JSON-serializable format"""
    result = []

    for message in messages:
        if isinstance(message, dict):
            # Already a dict, just add it
            result.append(message)
        else:
            # Message object, extract attributes
            message_dict = {
                "type": getattr(
                    message,
                    "type",
                    type(message).__name__.replace("Message", "").lower(),
                ),
                "content": getattr(message, "content", None),
                "id": getattr(message, "id", None),
            }

            if hasattr(message, "additional_kwargs"):
                message_dict["additional_kwargs"] = message.additional_kwargs

            if hasattr(message, "response_metadata"):
                message_dict["response_metadata"] = message.response_metadata

            if hasattr(message, "name"):
                message_dict["name"] = message.name

            if hasattr(message, "tool_calls"):
                message_dict["tool_calls"] = message.tool_calls

            if hasattr(message, "usage_metadata"):
                message_dict["usage_metadata"] = message.usage_metadata

            if hasattr(message, "tool_call_id"):
                message_dict["tool_call_id"] = message.tool_call_id

            result.append(message_dict)

    return result


async def event_generator(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}

    sent_message_count = 0

    for chunk in supervisor.stream(
        {
            "messages": [HumanMessage(content=request.user_input)],
            "generated_image": None,
        },
        config=config,
        stream_mode="updates",
    ):
        node_name = list(chunk)[0]
        node_update = chunk[node_name]

        print(f"Node update keys: {list(node_update.keys())}")
        generated_image = node_update.get("generated_image")

        all_messages = chunk[node_name]["messages"]
        new_messages = all_messages[sent_message_count:]

        if not new_messages and not generated_image:
            continue

        print(25 * "+")
        print("node name: ", node_name)
        print(f"New messages ({len(new_messages)}):", new_messages)
        if generated_image:
            print("Generated image present:", bool(generated_image))
        print(25 * "-")

        event_data = {
            "node_name": node_name,
            "messages": messages_to_json(new_messages),
            "generated_image": generated_image,
        }

        sent_message_count = len(all_messages)

        yield json.dumps(event_data, indent=2) + "\n"


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    return StreamingResponse(
        event_generator(request=request),
        media_type="text/event-stream",
    )


@app.get("/")
def read_root():
    return {"status": "RAG API is running, View at queai.vercel.app"}

import pprint
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from graph_builder import queai_graph
import json
import dotenv
from pydantic import BaseModel


dotenv.load_dotenv()


class ChatRequest(BaseModel):
    user_input: str
    thread_id: str | None = None


app = FastAPI()


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    user_input = request.user_input
    thread_id = request.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    def event_generator():
        sent_ids = set()
        final_answer_started = False

        events = queai_graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()
            # Yield tool call initiation info from AIMessage.tool_calls
            chatbot_msgs = event.get("chatbot", {}).get("messages", [])
            for cm in chatbot_msgs:
                # Yield info about each tool called by this AI message
                tool_calls = (
                    getattr(cm, "tool_calls", []) if hasattr(cm, "tool_calls") else []
                )
                for t in tool_calls:
                    tool_call_id = t.get("id", "")
                    if tool_call_id and tool_call_id not in sent_ids:
                        sent_ids.add(tool_call_id)
                        yield (
                            json.dumps(
                                {
                                    "type": "tool_call",
                                    "tool_name": t.get("name", ""),
                                    "args": t.get("args", {}),
                                    "tool_call_id": tool_call_id,
                                }
                            )
                            + "\n"
                        )

            # Yield tool results separately
            tools_msgs = event.get("tools", {}).get("messages", [])
            for tm in tools_msgs:
                tm_id = getattr(tm, "id", None) or tm.content
                if tm_id not in sent_ids:
                    sent_ids.add(tm_id)
                    yield (
                        json.dumps(
                            {
                                "type": "tool_result",
                                "tool_name": getattr(tm, "name", ""),
                                "tool_call_id": getattr(tm, "tool_call_id", ""),
                                "content": tm.content,
                            }
                        )
                        + "\n"
                    )

            # Yield chatbot intermediate/final messages
            for cm in chatbot_msgs:
                cm_id = getattr(cm, "id", None) or cm.content
                if cm_id not in sent_ids:
                    sent_ids.add(cm_id)
                    finish_reason = (
                        getattr(cm, "response_metadata", {}).get("finish_reason", None)
                        if hasattr(cm, "response_metadata")
                        else None
                    )
                    if finish_reason == "STOP":
                        final_answer_started = True
                        yield (
                            json.dumps({"type": "final_answer", "content": cm.content})
                            + "\n"
                        )
                    else:
                        if not final_answer_started and cm.content.strip():
                            yield (
                                json.dumps(
                                    {"type": "chatbot_subtask", "content": cm.content}
                                )
                                + "\n"
                            )

    return StreamingResponse(event_generator(), media_type="application/json")


@app.get("/")
def read_root():
    return {"status": "RAG API is running, View at queai.vercel.app"}


# Send History along with user input
# @app.post("/chat")
# async def chat_endpoint(request: ChatRequest):
#     final_state = None
#     for event in queai_graph.stream(
#         {
#             "messages": [{"role": "user", "content": request.user_input}],
#         }
#     ):
#         final_state = event
#
#     last_message = final_state[list(final_state.keys())[0]]["messages"][-1]
#     return {"responses": [last_message.content]}

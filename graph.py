from uuid import uuid4
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, MessagesState, StateGraph
from agents.supervisor_agent import supervisor_agent
from agents.code_agent import code_agent
from agents.research_agent import research_agent
from agents.image_agent import image_agent
from utils import pretty_print_messages
import sqlite3
import dotenv

dotenv.load_dotenv()

sqlite_connection = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_connection)

supervisor = (
    StateGraph(MessagesState)
    .add_node(
        supervisor_agent(),
        destinations=("research_agent", "code_agent", "image_agent", END),
    )
    .add_node(research_agent())
    .add_node(code_agent())
    .add_node(image_agent)
    .add_edge(START, "supervisor")
    .add_edge("research_agent", "supervisor")
    .add_edge("code_agent", "supervisor")
    .add_edge("image_agent", "supervisor")
    .compile(checkpointer=memory)
)

if __name__ == "__main__":
    while True:
        input_query = input("Enter prompt: ")
        config = {"configurable": {"thread_id": str(uuid4())}}

        for chunk in supervisor.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": input_query,
                    }
                ]
            },
            config=config,
        ):
            pretty_print_messages(chunk)

        final_message_history = chunk["supervisor"]["messages"]
        for message in final_message_history:
            message.pretty_print()

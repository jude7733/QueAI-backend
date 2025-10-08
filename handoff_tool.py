from typing import Annotated

from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import MessagesState
from langgraph.prebuilt import InjectedState
from langgraph.types import Command


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        task: str,
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}: {task}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,
            update={**state, "messages": state["messages"] + [tool_message]},
            graph=Command.PARENT,
        )

    return handoff_tool


assign_to_research_agent = create_handoff_tool(
    agent_name="research_agent",
    description="Assign a research task to the research agent.",
)

assign_to_coder_agent = create_handoff_tool(
    agent_name="code_agent",
    description="Assign a coding task to the python program executable agent.",
)

assign_to_image_agent = create_handoff_tool(
    agent_name="image_agent",
    description="Assign a image generation task to the image agent.",
)

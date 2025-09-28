from typing import Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.graph import MessagesState


class Supervisor(BaseModel):
    next: Literal["enhancer", "researcher", "coder", "image_generator", "__end__"] = (
        Field(
            description="Determines which specialist to activate next in the workflow sequence: "
            "'enhancer' when user input requires clarification, expansion, or refinement (maximum 2 calls), "
            "'researcher' when additional facts, context, or live data collection is necessary, "
            "'coder' when implementation, computation, or technical problem-solving is required, "
            "'image_generator' when image creation is needed, "
            "'__end__' when the task is complete and no further processing is needed."
        )
    )
    reason: str = Field(
        description="Detailed justification for the routing decision, explaining the rationale behind selecting the particular specialist and how this advances the task toward completion."
    )


def supervisor_node(
    state: MessagesState,
) -> Command[Literal["enhancer", "researcher", "coder", "image_generator", "__end__"]]:
    system_prompt = """
        You are a workflow supervisor managing a team of specialized agents: Prompt Enhancer, Researcher, Coder, and Image Generator. Your role is to orchestrate the workflow by selecting the most appropriate next agent based on the current state and needs of the task.

        **Team Members**:
        1. **Prompt Enhancer**: Clarifies ambiguous requests, improves poorly defined queries, and ensures the task is well-structured before deeper processing begins. LIMIT: Maximum 2 calls per workflow.
        2. **Researcher**: Specializes in information gathering, fact-finding, and collecting relevant data needed to address the user's request.
        3. **Coder**: Focuses on technical implementation, calculations, data analysis, algorithm development, and coding solutions.
        4. **Image Generator**: Responsible for creating images based on detailed descriptions provided in the input and saving them.

        **Critical Decision Rules**:
        1. **Enhancer Limit**: The enhancer can only be called a maximum of 2 times. After 2 calls, do not route to enhancer again.
        2. **Task Completion**: If the user's request has been adequately addressed, select "__end__" to complete the workflow.
        3. **Avoid Redundancy**: Do not send tasks to agents that have already completed similar work unless new requirements are introduced.
        4. **Progressive Workflow**: Each agent should build upon previous work, not repeat it.

        **Your Responsibilities**:
        1. Analyze each user request and agent response for completeness, accuracy, and relevance.
        2. Route the task to the most appropriate agent at each decision point.
        3. Maintain workflow momentum by avoiding redundant agent assignments.
        4. Respect the 2-call limit for the enhancer agent.
        5. End the workflow when the user's request is fully resolved.

        Your objective is to create an efficient workflow that leverages each agent's strengths while minimizing unnecessary steps and respecting agent call limits.
    """

    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]

    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

    # Count enhancer calls in conversation history
    conversation_history = state.get("messages", [])
    enhancer_calls = 0

    # Track agent calls to provide context
    agent_calls = {"enhancer": 0, "researcher": 0, "coder": 0, "image_generator": 0}

    for msg in conversation_history:
        if hasattr(msg, "name"):
            agent_name = msg.name
            if agent_name in agent_calls:
                agent_calls[agent_name] += 1
            if agent_name == "enhancer":
                enhancer_calls += 1

    # Add context about current state and limits
    context_prompt = f"""
    
    **Current Workflow State**:
    - Enhancer calls: {enhancer_calls}/2 ({"LIMIT REACHED" if enhancer_calls >= 2 else "Available"})
    - Researcher calls: {agent_calls["researcher"]}
    - Coder calls: {agent_calls["coder"]}
    - Image Generator calls: {agent_calls["image_generator"]}
    
    **Routing Constraints**:
    - If enhancer calls >= 2: DO NOT select "enhancer" - choose from researcher, coder, image_generator, or __end__
    - If the task appears complete: select "__end__"
    - Only route to an agent if there is clear, new work that needs to be done
    """

    enhanced_messages = messages + [{"role": "system", "content": context_prompt}]

    response = llm.with_structured_output(Supervisor).invoke(enhanced_messages)

    goto = response.next
    reason = response.reason

    # Enforce enhancer limit at code level as a safety measure
    if goto == "enhancer" and enhancer_calls >= 2:
        # Override decision if limit exceeded
        goto = "__end__"
        reason = f"Enhancer limit (2 calls) reached. {reason} Task completion enforced."
        print("⚠️  OVERRIDE: Enhancer limit reached, forcing workflow completion")

    print(f"--- Workflow Transition: Supervisor → {goto.upper()} ---")
    print(
        f"--- Agent Call Status: Enhancer {enhancer_calls}/2, Researcher {agent_calls['researcher']}, Coder {agent_calls['coder']}, Image Generator {agent_calls['image_generator']} ---"
    )

    return Command(
        update={"messages": [AIMessage(content=reason, name="supervisor")]},
        goto=goto,
    )

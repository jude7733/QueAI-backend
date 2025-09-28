from typing import Literal
from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.types import Command
from custom_tools.image_tool import generate_image_tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
import base64
from PIL import Image
from io import BytesIO
import os


def image_generator_node(state) -> Command[Literal["__end__"]]:
    """Generates images based on detailed descriptions provided in the input."""

    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    image_agent = create_react_agent(
        llm,
        tools=[generate_image_tool],
        prompt=(
            "You are an image generator. Create images based on detailed descriptions provided in the input."
        ),
    )

    result = image_agent.invoke(state)

    processed_images = []

    last_message = result["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_response = last_message.tool_calls[0]

        if hasattr(tool_response, "candidates"):
            for candidate in tool_response.candidates:
                for part in candidate.content.parts:
                    if part.inline_data is not None:
                        image_data = part.inline_data.data
                        image = Image.open(BytesIO(image_data))

                        import uuid

                        filename = f"generated_image_{uuid.uuid4()}.png"
                        filepath = os.path.join("images", filename)

                        os.makedirs("images", exist_ok=True)

                        image.save(filepath)

                        base64_image = base64.b64encode(image_data).decode("utf-8")

                        processed_images.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                                "filepath": filepath,
                            }
                        )

    content = "Generated images successfully."
    if processed_images:
        content = processed_images

    print("--- Workflow Transition: Image Generator → END ---")

    return Command(
        update={"messages": [AIMessage(content=content, name="image_generator")]},
        goto=END,
    )

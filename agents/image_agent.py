from langchain_core.messages import HumanMessage
from custom_tools.image_tool import generate_image
import dotenv

dotenv.load_dotenv()


def image_agent(state):
    messages = state["messages"]
    last_message = messages[-1].content

    if "image" in last_message.lower():
        prompt = (
            last_message.replace("generate image", "")
            .replace("create image", "")
            .strip()
        )

        result = generate_image(prompt)

        if result["success"]:
            # Add image data to state
            updated_state = {
                "messages": [
                    HumanMessage(
                        content=f"Image generated successfully for prompt: {prompt}"
                    )
                ],
                "generated_image": {
                    "data": result["image_data"],
                    "mime_type": result["mime_type"],
                    "prompt": result["prompt"],
                    "file_path": result["image_path"],
                },
            }
            return updated_state
        else:
            return {
                "messages": [
                    HumanMessage(content=f"Image generation failed: {result['error']}")
                ],
                "generated_image": None,
            }

    return {
        "messages": [HumanMessage(content="No image generation requested.")],
        "generated_image": None,
    }


if __name__ == "__main__":
    test_state = {"messages": [HumanMessage(content="generate of a cartoon plane")]}
    result = image_agent(test_state)
    print("Result:", result)

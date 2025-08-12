import os
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from langchain_core.tools import tool
import dotenv

dotenv.load_dotenv()


@tool(
    description="Generates an image based on the provided prompt.",
    args_schema={"prompt": str},
)
def generate_image_tool(prompt: str) -> types.GenerateContentResponse:
    """
    Image generation tool using Google Gemini API.
    """
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )

    return response


if __name__ == "__main__":
    response = generate_image_tool(
        "Hi, can you create a 3d rendered image of a pig with wings and a top hat flying over a happy futuristic scifi city with lots of greenery?"
    )
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO((part.inline_data.data)))
            image.show()

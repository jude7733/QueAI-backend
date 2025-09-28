from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from langchain_core.tools import tool
import dotenv
from pydantic import BaseModel
import base64

dotenv.load_dotenv()


class ImageGenInput(BaseModel):
    prompt: str


@tool(
    description="Generates an image based on the provided prompt.",
    args_schema=ImageGenInput,
)
def generate_image_tool(prompt: str):
    """
    Image generation tool using Google Gemini API.
    Returns base64 encoded image data in LangGraph compatible format.
    """
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",  # Use the experimental model for image generation
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )

    images_data = []

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            continue
        elif part.inline_data is not None:
            # Convert image data to base64
            image_data = part.inline_data.data
            base64_image = base64.b64encode(image_data).decode("utf-8")

            images_data.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

    if images_data:
        return images_data
    else:
        return "No images were generated."


@tool(
    description="Generates an image and saves it to disk, returning the file path.",
    args_schema=ImageGenInput,
)
def generate_image_and_save_tool(prompt: str) -> str:
    """
    Image generation tool that saves images to disk.
    Returns file paths of saved images.
    """
    import os
    import uuid

    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )

    saved_files = []

    os.makedirs("generated_images", exist_ok=True)

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            image_data = part.inline_data.data
            image = Image.open(BytesIO(image_data))

            filename = f"generated_image_{uuid.uuid4()}.png"
            filepath = os.path.join("generated_images", filename)

            image.save(filepath)
            saved_files.append(filepath)

    if saved_files:
        return (
            f"Generated and saved {len(saved_files)} image(s): {', '.join(saved_files)}"
        )
    else:
        return "No images were generated."

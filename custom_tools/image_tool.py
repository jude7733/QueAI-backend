from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from langchain_core.tools import tool
import dotenv
from pydantic import BaseModel

dotenv.load_dotenv()


class ImageGenInput(BaseModel):
    prompt: str


@tool(
    description="Generates an image based on the provided prompt.",
    args_schema=ImageGenInput,
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
    response = generate_image_tool.invoke(
        {
            "prompt": "Generate a high-resolution, photorealistic portrait of a young woman sitting at a rustic wooden table in a sunlit café, with natural lighting, detailed skin texture, realistic hair, and soft depth-of-field background."
        }
    )
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO((part.inline_data.data)))
            image.show()

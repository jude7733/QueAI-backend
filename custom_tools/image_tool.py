import os
import base64
from uuid import uuid4
from google import genai


def generate_image(prompt: str) -> dict:
    try:
        os.makedirs("generated_images", exist_ok=True)

        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"]
            ),
        )

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if (
                    hasattr(part, "inline_data")
                    and part.inline_data
                    and part.inline_data.mime_type == "image/png"
                ):
                    # Get binary data
                    image_data = part.inline_data.data

                    # Save to file
                    image_path = f"generated_images/image_{uuid4()}.png"
                    with open(image_path, "wb") as f:
                        f.write(image_data)

                    # Convert to base64
                    image_b64 = base64.b64encode(image_data).decode("utf-8")

                    return {
                        "success": True,
                        "image_data": image_b64,
                        "image_path": image_path,
                        "mime_type": "image/png",
                        "prompt": prompt,
                    }

        return {"success": False, "error": "No image found in response"}

    except Exception as e:
        return {"success": False, "error": str(e)}

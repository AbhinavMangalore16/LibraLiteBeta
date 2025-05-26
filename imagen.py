import base64
import os
from dotenv import load_dotenv

# Load .env and Google API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize Gemini with image generation
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-preview-image-generation")

message = {
    "role": "user",
    "content": "Generate a photorealistic image of a cuddly cat wearing a hat.",
}

# Invoke the model with text + image generation
response = llm.invoke(
    [message],
    generation_config=dict(response_modalities=["TEXT", "IMAGE"]),
)

# Extract image base64
def _get_image_base64(response: AIMessage) -> str:
    image_block = next(
        block
        for block in response.content
        if isinstance(block, dict) and block.get("image_url")
    )
    return image_block["image_url"].get("url").split(",")[-1]

# Decode and save
image_base64 = _get_image_base64(response)
image_data = base64.b64decode(image_base64)

# Save image to file
output_path = "generated_cat.jpg"
with open(output_path, "wb") as f:
    f.write(image_data)

print(f"Image saved to {output_path}")

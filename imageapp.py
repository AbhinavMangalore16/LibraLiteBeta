import os
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-preview-image-generation")

# Function to extract base64 image from response
def _get_image_base64(response: AIMessage) -> str:
    image_block = next(
        block
        for block in response.content
        if isinstance(block, dict) and block.get("image_url")
    )
    return image_block["image_url"].get("url").split(",")[-1]

# Streamlit UI
st.title("🖼️ Gemini Image Generator")

prompt = st.text_input("Enter your prompt:", "A photorealistic image of a cuddly cat wearing a hat")

if st.button("Generate Image"):
    with st.spinner("Generating..."):
        message = {"role": "user", "content": prompt}
        response = llm.invoke(
            [message],
            generation_config=dict(response_modalities=["TEXT", "IMAGE"]),
        )

        try:
            image_base64 = _get_image_base64(response)
            image_data = base64.b64decode(image_base64)
            st.image(image_data, caption="Generated Image", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to generate image: {e}")

import os
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

story_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4, max_retries=2)
image_llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-preview-image-generation")

title_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative assistant."),
    ("human", "Create a concise story title from this summary: '{summary}'. Return only the title.")
])
char_prompt = ChatPromptTemplate.from_messages([
    ("system", "You create main characters."),
    ("human", "For the story titled '{title}', list main characters with name, age, personality, and background.")
])
story_prompt = ChatPromptTemplate.from_messages([
    ("system", "You write stories."),
    ("human", "Write a long deep story titled '{title}' featuring these characters:\n\n{characters}")
])

title_chain = title_prompt | story_llm
char_chain = RunnableLambda(lambda res: {"title": res.content.strip()}) | char_prompt | story_llm
story_chain = RunnableLambda(lambda inp: {"title": inp["title"], "characters": inp["characters"]}) | story_prompt | story_llm

def get_image_base64(response: AIMessage) -> str:
    block = next(b for b in response.content if isinstance(b, dict) and b.get("image_url"))
    return block["image_url"]["url"].split(",")[-1]

st.set_page_config(page_title="Libra Lite", layout="centered")
st.title("✨ Libra Lite -  β version✨")

summary = st.text_area("Enter story summary:", height=150)

if st.button("Generate"):
    if not summary.strip():
        st.warning("Please enter a summary.")
    else:
        with st.spinner("Generating..."):
            # Generate title
            title_res = title_chain.invoke({"summary": summary})
            title = title_res.content.strip()
            # Generate characters
            char_res = char_chain.invoke(title_res)
            characters = char_res.content.strip()
            # Generate story
            story_res = story_chain.invoke({"title": title, "characters": characters})
            story = story_res.content.strip()
            # Generate image prompt and get image
            prompt = f"Photorealistic illustration of: {title}"
            img_msg = {"role": "user", "content": prompt}
            img_res = image_llm.invoke([img_msg], generation_config={"response_modalities": ["TEXT", "IMAGE"]})

            try:
                img_b64 = get_image_base64(img_res)
                img_data = base64.b64decode(img_b64)
            except Exception:
                img_data = None

        if img_data:
            st.image(img_data, use_column_width=True)
        st.balloons()
        st.subheader("📝 Title")
        st.success(title)

        st.subheader("🎭 Characters")
        st.info(characters)

        st.subheader("📖 Story")
        st.write(story)


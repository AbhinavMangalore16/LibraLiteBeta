import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# Load .env and Google API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.4,
    max_retries=2
)

# --- Prompt Templates ---
title_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative assistant."),
    ("human", "Given the summary: '{summary}', create a title for a short story. Only return a single title.")
])

char_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a great casting director."),
    ("human", "Given the story title: '{title}', create main characters with name, age, personality trait, and background.")
])

story_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a great fiction author."),
    ("human", "Write a short story titled '{title}' featuring the following characters:\n\n{characters}")
])

# --- Chains ---
title_chain = title_prompt | llm

char_chain = (
    RunnableLambda(lambda title_output: {"title": title_output.content.strip()})
    | char_prompt
    | llm
)

story_chain = (
    RunnableLambda(lambda inputs: {
        "title": inputs["title"],
        "characters": inputs["characters"]
    })
    | story_prompt
    | llm
)

# --- Streamlit UI ---
st.set_page_config(page_title="🧠 Gemini Story Generator", layout="wide")
st.title("🧠 Gemini-Powered Story Generator")

summary_input = st.text_area("📌 Enter your story summary", height=150)

if st.button("🚀 Generate Story"):
    if not summary_input.strip():
        st.warning("Please enter a story summary to continue.")
    else:
        with st.spinner("Generating story..."):
            # Step 1: Title
            title_result = title_chain.invoke({"summary": summary_input})
            title = title_result.content.strip()

            # Step 2: Characters
            characters_result = char_chain.invoke(title_result)
            characters = characters_result.content.strip()

            # Step 3: Story
            story_result = story_chain.invoke({
                "title": title,
                "characters": characters
            })
            story = story_result.content.strip()

        # Display results
        st.subheader("📝 Title")
        st.success(title)

        st.subheader("🎭 Characters")
        st.info(characters)

        st.subheader("📖 Story")
        st.write(story)

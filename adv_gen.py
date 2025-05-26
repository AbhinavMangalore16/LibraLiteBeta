import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.4,
    max_retries=2
)

# --- Prompts ---
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
# Step 1: summary → title
title_chain = title_prompt | llm

# Step 2: title → characters
char_chain = (
    RunnableLambda(lambda title_output: {"title": title_output.content.strip()})
    | char_prompt
    | llm
)

# Step 3: (title + characters) → story
story_chain = (
    RunnableLambda(lambda char_output_and_title: {
        "title": char_output_and_title["title"],
        "characters": char_output_and_title["characters"]
    })
    | story_prompt
    | llm
)

# --- Execution Function ---
def generate_full_story(summary: str):
    # Step 1: Generate title
    title_output = title_chain.invoke({"summary": summary})
    title = title_output.content.strip()
    print("\n📝 Title:")
    print(title)

    # Step 2: Generate characters
    characters_output = char_chain.invoke(title_output)
    characters = characters_output.content.strip()
    print("\n🎭 Characters:")
    print(characters)

    # Step 3: Generate story
    story_output = story_chain.invoke({
        "title": title,
        "characters": characters
    })
    print("\n📖 Story:")
    print(story_output.content.strip())


# --- Run It ---
generate_full_story("In a world where memories can be traded, a young girl sells hers to save her brother.")

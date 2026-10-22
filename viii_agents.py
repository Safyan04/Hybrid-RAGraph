from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from vi_tools import all_tools
from vii_prompts import SYSTEM_PROMPT

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import ModelFallbackMiddleware

load_dotenv()

groq_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)   # list mein na ho to openai/gpt-oss-20b
genai_llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

agent = create_agent(
    model=groq_llm,
    tools=all_tools,
    system_prompt=SYSTEM_PROMPT,
    middleware=[ModelFallbackMiddleware(genai_llm)],
    checkpointer=InMemorySaver(),
)
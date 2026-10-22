from datetime import datetime
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import numexpr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from ii_chunking import chunks
from v_retrieval import retriever


@tool(response_format="content_and_artifact")
def document_search(query: str):
    """ALWAYS use this first. Searches the uploaded document and returns relevant passages with their page numbers."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant passages found.", []
    text = "\n\n".join(
        f"[Page {int(d.metadata.get('page', 0)) + 1}]\n{d.page_content}" for d in docs
    )
    return text, docs   # text LLM ke liye, docs app ke sources ke liye

rag_tool = document_search

@tool
def get_datetime() -> str:
    """Returns the current date and time. Use when the user asks about today's date or time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculator(expression: str) -> str:
    """Evaluates a math expression, e.g. '(25*4)+10/2'. Use for any calculation."""
    try:
        return str(numexpr.evaluate(expression))
    except Exception as e:
        return f"Invalid expression: {e}"

ddg = DuckDuckGoSearchResults(num_results=3)

@tool
def web_search(query: str) -> str:
    """Search the internet for general knowledge, recent news, or anything not in the document."""
    try:
        return ddg.invoke(query)
    except Exception as e:
        return f"Web search failed ({type(e).__name__})."

@tool
def list_sources() -> str:
    """Lists all document files/sources available in the knowledge base."""
    sources = {c.metadata.get("source", "unknown") for c in chunks}
    return "\n".join(sorted(sources))

all_tools = [rag_tool, get_datetime, calculator, web_search, list_sources]
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from ii_chunking import chunks

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model='gemini-embedding-2'
    )

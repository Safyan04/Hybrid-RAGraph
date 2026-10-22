from langchain_text_splitters import RecursiveCharacterTextSplitter
from i_loading import doc

loader = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100)

chunks = loader.split_documents(doc)
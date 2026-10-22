from langchain_community.vectorstores import Chroma
from i_loading import doc
from iii_embedding import embedding

COLLECTION_NAME = 'ABC'
PERSIST_direc = 'hyb'

vector = Chroma.from_documents(
                documents=doc,
                embedding=embedding,
                collection_name=COLLECTION_NAME,
                persist_directory=PERSIST_direc
                )

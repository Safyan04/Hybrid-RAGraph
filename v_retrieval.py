from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from ii_chunking import chunks
from iv_vectordb import vector

vect_retriever = vector.as_retriever(
    search_type= 'mmr',
    search_kwargs = {'k':3}
)

bm_retriever = BM25Retriever.from_documents(chunks)
bm_retriever.k = 3

retriever = EnsembleRetriever(
    retrievers=[bm_retriever, vect_retriever],
    weights=[0.4, 0.6]
)
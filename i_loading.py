from langchain_community.document_loaders import PyPDFLoader

pdf = PyPDFLoader("uploaded.pdf")

doc = pdf.load()
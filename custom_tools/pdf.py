from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def load_pdf_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def split_documents(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_documents(pages)


def create_vector_store(split_docs):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    document_texts = [doc.page_content for doc in split_docs]

    embeddings = embedder.encode(document_texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    return index, document_texts, embedder


def retrieve_context(query, embedder, index, documents, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding.astype(np.float32), k)
    return [documents[i] for i in indices[0]]


def generate_answer(query, context):
    formatted_context = "\n".join(context)

    prompt = f"""You are an export assistant trained on document information. Use this context to answer the question:

    {formatted_context}

    Question: {query}

    Answer in detail using only the provided context. """

    response = ollama.generate(
        model="deepseek-1.5B-npu",
        prompt=prompt,
        options={
            "temperature": 0.3,
            "max_new_tokens": 2000,
        },
    )

    return response["response"]

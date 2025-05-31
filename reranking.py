from helper_utils import word_wrap, load_chroma
from pypdf import PdfReader
import os
import huggingface_hub
from dotenv import load_dotenv


from pypdf import PdfReader
import numpy as np

from langchain_community.document_loaders import PyPDFLoader


# Load environment variables from .env file
load_dotenv()

hf_key = os.getenv("HUGGINGFACE_API_KEY")
client = huggingface_hub.InferenceClient(api_key=hf_key)


import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction()


reader = PdfReader("data/Stories Of The Prophets By Ibn Kathir.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter the empty strings
pdf_texts = [text for text in pdf_texts if text]

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(
    "prophets-collect", embedding_function=embedding_function
)

# extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_texts))]

chroma_collection.add(ids=ids, documents=token_split_texts)

count = chroma_collection.count()

query = "Why all prophets were Muslims?"

results = chroma_collection.query(
    query_texts=query, n_results=10, include=["documents", "embeddings"]
)

retrieved_documents = results["documents"][0]

for document in results["documents"][0]:
    print(word_wrap(document))
    print("")

from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [[query, doc] for doc in retrieved_documents]
scores = cross_encoder.predict(pairs)

print("Scores:")
for score in scores:
    print(score)

print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o + 1)

original_query = (
    "What are the common virtues and characteristics shared by all prophets?"
)

generated_queries = [
    "How did prophets demonstrate patience and perseverance in their missions?",
    "What moral qualities were emphasized in the stories of the prophets?",
    "How did the prophets communicate Allah's message to their people?",
    "What trials and challenges did prophets typically face in their missions?",
    "How did prophets exemplify faith and submission to Allah's will?",
]

# concatenate the original query with the generated queries
queries = [original_query] + generated_queries


results = chroma_collection.query(
    query_texts=queries, n_results=10, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"]

# Deduplicate the retrieved documents
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)

unique_documents = list(unique_documents)

pairs = []
for doc in unique_documents:
    pairs.append([original_query, doc])

scores = cross_encoder.predict(pairs)

print("Scores:")
for score in scores:
    print(score)

print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o)
# ====
top_indices = np.argsort(scores)[::-1][:5]
top_documents = [unique_documents[i] for i in top_indices]

# Concatenate the top documents into a single context
context = "\n\n".join(top_documents)


# Generate the final answer using the Hugging Face model
def generate_multi_query(query, context, model="tinydolphin"):

    prompt = f"""
    You are a knowledgeable Islamic  scholar and historian. 
    Your users are inquiring about the Stories Of The Prophets By Ibn Kathir. 
    """

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": f"based on the following context:\n\n{context}\n\nAnswer the query: '{query}'",
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response
    content = content.split("\n")
    return content


res = generate_multi_query(query=original_query, context=context)
print("Final Answer:")
print(res)

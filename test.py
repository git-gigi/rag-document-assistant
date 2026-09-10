from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load the existing database
db = Chroma(persist_directory="./chroma_db_dir", embedding_function=embeddings)

# Ask a "technical" question
query = "Cos'è la firma digitale qualificata?"
risultati = db.similarity_search(query, k=2) # k=2 takes the top 2 fragments

for doc in risultati:
    print(f"\n--- Trovato in: {doc.metadata['origin']} ---")
    print(doc.page_content)
#!/usr/bin/env python3
import sys
sys.path.append('.')
from src.retrieval.vector_store import ChromaDBVectorStore

# Initialize the vector store
store = ChromaDBVectorStore()

# Get collection info
total_docs = store.collection.count()
print(f"Total documents in ChromaDB: {total_docs}")

# Get some sample documents
if total_docs > 0:
    results = store.collection.get(limit=5)
    print(f"\nFirst {min(5, total_docs)} documents:")
    for i, (doc_id, content, metadata) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
        print(f"\n{i+1}. ID: {doc_id}")
        print(f"   Content preview: {content[:100]}...")
        print(f"   Metadata: {metadata}")
else:
    print("\nNo documents found in the database!")
    
# Check if we can query
from src.embedding.embedder import SentenceTransformerEmbedder
embedder = SentenceTransformerEmbedder(model_name='all-MiniLM-L6-v2')

test_query = "What is BeachBox?"
print(f"\n\nTesting query: '{test_query}'")
query_embedding = embedder.embed([test_query])[0]

# Search for similar documents
results = store.search(query_embedding, k=3)
print(f"\nSearch results: {len(results)} documents found")
for i, result in enumerate(results):
    print(f"\n{i+1}. Distance: {result.get('distance', 'N/A')}")
    print(f"   Content preview: {result.get('content', '')[:100]}...")
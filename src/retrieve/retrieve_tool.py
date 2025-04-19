import os
import sys
import json
import torch
import numpy as np

from typing import List, Dict, Any
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from astrapy import DataAPIClient, Database, Collection
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from transformers import AutoTokenizer, AutoModel
# from sklearn.metrics.pairwise import cosine_similarity

from old import vi_retrieve


# Load environment variables
load_dotenv()

# Configure Google API
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=GOOGLE_API_KEY)
except ValueError as e: 
    print(f"Configuration Error: {e}")
    print("Please set the GOOGLE_API_KEY environment variable.")
    sys.exit(1)
except Exception as e: 
    print(f"An unexpected error occurred during configuration: {e}")
    sys.exit(1)
    
def connect_to_datastax() -> Database:
    endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")  
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    if not token or not endpoint:
        raise RuntimeError(
            "Environment variables ASTRA_DB_API_ENDPOINT and ASTRA_DB_APPLICATION_TOKEN must be defined"
        )
    
    # Create an instance of the DataAPIClient
    client = DataAPIClient()
   
    # Connect to database
    database = client.get_database(endpoint, token=token)
    print(f"Connected to database {database.info().name}")
    return database

def connect_to_collection(database: Database, collection_name: str) -> Collection:
    try:
        collection = database.get_collection(collection_name)
        print(f"Connected to collection {collection_name}")
        return collection
    except Exception as e:
        print(f"Error connecting to collection {collection_name}: {e}")
        raise


def generate_question_embedding(question: str, dimensions: int = 768) -> List[float]:
    try:
        # Initialize embeddings model for queries
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            # task_type="retrieval_document",
            task_type="semantic_similarity"

        )
        
        # Generate embedding
        embedding = embeddings_model.embed_documents(question)
        
        # Verify dimensions
        if len(embedding) != dimensions:
            print(f"Warning: Expected {dimensions} dimensions but got {len(embedding)}")
            
        return embedding
    
    except Exception as e:
        print(f"Error generating embedding for question: {e}")
        raise

def search_similar_vector(
    collection: Collection,
    question_embedding: List[float],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    try:
        results = collection.find(
            {},
            sort={"$vector": question_embedding},
            limit=top_k,
            include_similarity=True,   
        )
        return results
    except Exception as e:
        print(f"Error searching for similar vectors: {e}")
        raise
    
def display_results(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No results found.")
        return
    for result in results:
        print(f" {result['$similarity']} {result['content']}")
        print("===========")
    
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
    model = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
    collection_name = "law_vi"
    
    query = 'không đội mũ bảo hiểm'
    embedding_query = vi_retrieve.embedding_query(query, tokenizer, model)
    embedding_query_list = embedding_query.flatten().tolist()

    print(embedding_query.shape)
    
    
    # Connect to database and collection
    try:
        database = connect_to_datastax()
        collection = connect_to_collection(database, collection_name)
    except Exception as e:
        print(f"Failed to connect to database or collection: {e}")
        sys.exit(1)
    
    results = search_similar_vector(collection,embedding_query_list, 5)
    display_results(results)
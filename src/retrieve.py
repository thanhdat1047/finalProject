import os
import sys
from typing import List, Dict, Any
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from astrapy import DataAPIClient, Database, Collection
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore

# Load environment variables
load_dotenv()

def astax():
    endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")  
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")  
    keyspace = os.getenv("ASTRA_DB_KEYSPACE")  
    
    embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            # task_type="retrieval_document",
            task_type="semantic_similarity")
    
    vstore = AstraDBVectorStore(
        collection_name='law2',
        embedding=embeddings_model,
        token=token,
        api_endpoint=endpoint,
        namespace=keyspace    
    )
    
    retriever = vstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 20, "score_threshold": 0.5},
    )
    
    results = retriever.invoke("hơi thở có nồng độ cồn")
    
    # results = vstore.similarity_search(" ", k=15, filter={"chuong":"chương 2. hành vi vi phạm, hình thức, mức xử phạt, mức trừ điểm giấy phép lái xe và biện pháp khắc phục hậu quả vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ", "muc":"mục 1. vi phạm quy tắc giao thông đường bộ"})
    
    for result in results:
        print(f" {result.page_content}")
        print("=======\n")

        

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
    """
    Connect to DataStax Astra DB database
    
    Returns:
        Database: Connected database object
    """
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

def search_similar_vectors(
    collection: Collection,
    question_embedding: List[float],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    try:
        # Use vector search API to find similar documents
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
    """
    Display the search results in a readable format
    
    Args:
        results: List of document results from vector search
    """
    if not results:
        print("No results found.")
        return
    for result in results:
        print(f" {result['$similarity']} {result['content']}")
        
    # print(f"\nFound {len(results)} relevant documents:")
    # for i, doc in enumerate(results, 1):
        # print(f"\n--- Result {i} ---")
        
        # # Display content if available
        # if 'content' in doc:
        #     content = doc['content']
        #     print(f"Content: {content[:200]}..." if len(content) > 200 else f"Content: {content}")
        
        # # Display similarity score if available
        # if '$similarity' in doc:
        #     print(f"Similarity: {doc['$similarity']:.4f}")
            
        # # Display metadata if available
        # if 'metadata' in doc:
        #     meta = doc['metadata']
        #     for key, value in meta.items():
        #         if value:  # Only display non-empty values
        #             print(f"{key}: {value}")

def main():
    # Define collection name
    collection_name = "law2"  # Update with your actual collection name
    
    # Connect to database and collection
    try:
        database = connect_to_datastax()
        collection = connect_to_collection(database, collection_name)
    except Exception as e:
        print(f"Failed to connect to database or collection: {e}")
        sys.exit(1)
    
    # Example usage - interactive mode
    print("\n--- Legal Document Retrieval System ---")
    print("Type 'exit' to quit")
    
    while True:
        # Get user question
        question = input("\nEnter your legal question: ").strip()
        
        if question.lower() == 'exit':
            break
        
        if not question:
            continue
        
        try:
            print("Generating embedding...")
            question_embedding = generate_question_embedding(question)
            
            print("Searching for similar documents...")
            results = search_similar_vectors(collection, question_embedding)
            
            # Display results
            display_results(results)
        
        except Exception as e:
            print(f"Error processing question: {e}")
    
    print("\n--- Session ended ---")

if __name__ == "__main__":
    # main()
    astax()
# import json 
# import os
# import sys
# import google.generativeai as genai
# from google.generativeai import types
# import time 
# import numpy as np
# from typing import List, Dict, Any

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_core.documents import Document
# from uuid import uuid4

# # Configure Google API
# try:
#     GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
#     if not GOOGLE_API_KEY:
#         raise ValueError("GOOGLE_API_KEY environment variable not set.")
#     genai.configure(api_key=GOOGLE_API_KEY)
# except ValueError as e: 
#     print(f"Configuration Error: {e}")
#     print("Please set the GOOGLE_API_KEY environment variable.")
#     sys.exit(1)
# except Exception as e: 
#     print(f"An unexpected error occurred during configuration: {e}")
#     sys.exit(1)


# def load_raw_data(filepath="./metadata/metadata.json"):
#     """
#     Loads raw data from a JSON file
#     """
#     try:
#         with open(filepath, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             if not isinstance(data, list):
#                 print(f"Error: Expected a JSON list in {filepath}, but got {type(data)}")
#                 return None
#             print(f"Successfully loaded {len(data)} raw entries from {filepath}")
#             return data
#     except FileNotFoundError:
#         print(f"Error: File not found at {filepath}")
#         return None
#     except Exception as e:
#         print(f"An unexpected error occurred while loading raw data: {e}")
#         return None
    

# def create_langchain_documents(raw_data):
#     if not raw_data:
#         print("Input raw_data is empty or None. Cannot create documents.")
#         return []
    
#     documents = []
#     print(f"Creating LangChain Documents from {len(raw_data)} raw entries...")
#     for i, chunk in enumerate(raw_data):
#         try:
#             # Ensure chunk is a dictionary
#             if not isinstance(chunk, dict):
#                 print(f"Warning: Skipping item at index {i} because it is not a dictionary: {chunk}")
#                 continue
            
#             # Concat fields, handling missing keys/None
#             chuong = chunk.get('chuong','') or ''
#             muc = chunk.get('muc', '') or ''
#             dieu = chunk.get('dieu', '') or ''
#             content = chunk.get('content', '') or ''
            
#             # Combine the fields with newline
#             page_content = f"{muc}.{dieu}.{content}".strip()
            
#             # Keep all fields
#             metadata = chunk.copy()
            
#             # Create langchain doc
#             if not page_content:
#                 print(f"Warning: Skipping empty page_content for chunk index {i}")
#                 continue
            
#             doc = Document(page_content=page_content, metadata=metadata)
#             documents.append(doc)
#         except Exception as e:
#             print(f"Error processing chunk at index {i}: {e}. Skipping this chunk.")
#             continue # Skip problematic chunks

#     print(f"Successfully created {len(documents)} LangChain Documents.")
#     return documents


# class NumpyEncoder(json.JSONEncoder):
#     """Special JSON encoder for numpy types"""
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)


# def generate_embeddings(documents: List[Document], output_file: str) -> None:
#     """
#     Generate embeddings for documents and save them to a JSON file
    
#     Args:
#         documents: List of LangChain Document objects
#         output_file: Path to save the JSON file with embeddings
#     """
#     print(f"Generating embeddings for {len(documents)} documents...")
    
#     # Initialize embeddings model
#     embeddings_model = GoogleGenerativeAIEmbeddings(
#         model="models/text-embedding-004",
#         # task_type="retrieval_document",
#         task_type="semantic_similarity"
#     )
    
#     # task_type=types.
#     # Result container
#     results = []
    
#     # Process in batches to avoid rate limits
#     batch_size = 5
#     for i in range(0, len(documents), batch_size):
#         batch_docs = documents[i:i+batch_size]
#         batch_ids = [str(uuid4()) for _ in range(len(batch_docs))]
        
#         # Get text from documents
#         texts = [doc.page_content for doc in batch_docs]
        
#         # Generate embeddings for the batch
#         try:
#             batch_embeddings = embeddings_model.embed_documents(texts)
            
#             # Combine all data
#             for j, (doc, embedding, doc_id) in enumerate(zip(batch_docs, batch_embeddings, batch_ids)):
#                 doc_entry = {
#                     "id": doc_id,
#                     "content": doc.page_content,
#                     "metadata": doc.metadata,
#                     "embedding": embedding
#                 }
#                 results.append(doc_entry)
                
#             print(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch_docs)} documents)")
            
#             # Sleep to avoid rate limiting
#             time.sleep(1)
            
#         except Exception as e:
#             print(f"Error generating embeddings for batch starting at index {i}: {e}")
    
#     # Save to JSON file
#     try:
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
#         print(f"Successfully saved {len(results)} document embeddings to {output_file}")
#     except Exception as e:
#         print(f"Error saving embeddings to file: {e}")


# # Main
# if __name__ == "__main__":
#     print("\n--- Starting Embedding Generation Process ---")
    
#     # Load data
#     json_filepath = "./metadata/metadata.json"
#     output_filepath = "./embeddings/legal_embeddings2.json"
    
#     # Create output directory if it doesn't exist
#     os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
#     # Process data and generate embeddings
#     raw_legal_data = load_raw_data(json_filepath)
    
#     if raw_legal_data:
#         langchain_docs = create_langchain_documents(raw_legal_data)
        
#         if langchain_docs:
#             generate_embeddings(langchain_docs, output_filepath)
#         else:
#             print("\nNo LangChain documents were created. Embedding process aborted.")
#     else:
#         print(f"\nCould not load data from {json_filepath}. Embedding process aborted.")

#     print("\n--- Embedding Generation Process Finished ---")
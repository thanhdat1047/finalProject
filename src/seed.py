import json 
import os
import sys;
import google.generativeai as genai
import time 

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.api_core import exceptions as google_exceptions
from langchain_core.documents import Document
from langchain_milvus import Milvus
from uuid import uuid4


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


def load_raw_data(filepath="./metadata/metadata.json"):
    """
    Loads raw data from a JSON file
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"Error: Expected a JSON list in {filepath}, but got {type(data)}")
                return None
            print(f"Successfully loaded {len(data)} raw entries from {filepath}")
            return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading raw data: {e}")
        return None
    

def create_langchain_documents(raw_data):
    if not raw_data:
        print("Input raw_data is empty or None. Cannot create documents.")
        return []
    
    documents = []
    print(f"Creating LangChain Documents from {len(raw_data)} raw entries...")
    for i, chunk in enumerate(raw_data):
        try:
            # Ensure chunk is a ditionary
            if not isinstance(chunk, dict):
                print(f"Warning: Skipping item at index {i} because it is not a dictionary: {chunk}")
                continue
            
            # Concat fields, handling missing keys/None
            chuong = chunk.get('chuong','') or ''
            muc = chunk.get('muc', '') or ''
            dieu = chunk.get('dieu', '') or ''
            content = chunk.get('content', '') or ''
            
            # Combine the fields with newline
            page_content = f"{content}".strip()
            
            # Keep all fields
            metadata = chunk.copy()
            
            #Create langchain doc
            if not page_content:
                print(f"Warning: Skipping empty page_content for chunk index {i}")
                continue
            
            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)
        except Exception as e:
            print(f"Error processing chunk at index {i}: {e}. Skipping this chunk.")
            continue # Skip problematic chunks

    print(f"Successfully created {len(documents)} LangChain Documents.")
    return documents
          
          
def seed_milvus(URI_link:str , collection_name: str, documents: Document) -> Milvus:
    # embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        task_type="retrieval_document",
    )
    
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri":URI_link},
        collection_name=collection_name,
        drop_old=True
    )
    
    uuids = [str(uuid4()) for _ in range(len(documents))]

    
    batch_size =5
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_ids = uuids[i:i+batch_size]
        vectorstore.add_documents(documents=batch_docs, ids = batch_ids)
        print(f"Added {len(batch_docs)} documents, sleeping for 1s...")
        time.sleep(1)  # Chờ 1 giây để tránh rate limit
        
    # vectorstore.add_documents(documents=documents,ids =uuids)
    print('vector: ', vectorstore)
    return vectorstore

def connect_to_milvus(URI_link: str, collection_name: str) -> Milvus:
    """
    Hàm kết nối đến collection có sẵn trong Milvus
    Args:
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection cần kết nối
    Returns:
        Milvus: Đối tượng Milvus đã được kết nối, sẵn sàng để truy vấn
    Chú ý:
        - Không tạo collection mới hoặc xóa dữ liệu cũ
        - Sử dụng model 'models/text-embedding-004' cho việc tạo embeddings khi truy vấn
    """
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        task_type="RETRIEVAL_QUERY"
    )

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,   
    )
    return vectorstore

# main
if __name__ == "__main__":

    print("\n--- Starting LangChain Embedding Process ---")
        
    # load data
    json_filepath = "./metadata/metadata.json"
    raw_legal_data = load_raw_data(json_filepath)
        
    if raw_legal_data:
        langchaindocs = create_langchain_documents(raw_legal_data)
            
        if langchaindocs:
            seed_milvus('http://localhost:19530','legal_data_2', langchaindocs)
        else:
            print("\nNo LangChain documents were created. Embedding process aborted.")
    else:
        print(f"\nCould not load data from {json_filepath}. Embedding process aborted.")

    print("\n--- LangChain Process Finished ---")
                                



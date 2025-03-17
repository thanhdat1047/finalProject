import os 
import getpass
import json
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_milvus import Milvus
from langchain.schema import Document
from dotenv import load_dotenv
from uuid import uuid4
from langchain_ollama import OllamaEmbeddings
import time 

load_dotenv()

def load_data_from_local(filename: str, directory: str) -> tuple: 
    """
    Hàm đọc dữ liệu từ file JSON local
    Args:
        filename (str): Tên file JSON cần đọc (ví dụ: 'data.json')
        directory (str): Thư mục chứa file (ví dụ: 'data_v3')
    Returns:
        tuple: Trả về (data, doc_name) trong đó:
            - data: Dữ liệu JSON đã được parse
            - doc_name: Tên tài liệu đã được xử lý (bỏ đuôi .json và thay '_' bằng khoảng trắng)
    """
    file_path = os.path.join(directory, filename)
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None, None
    
    with open(file_path, 'r',encoding='utf-8') as file: 
        data = json.load(file)
    print(f'Data loaded from {file_path}')
    return data, filename.rsplit('.',1)[0].replace('_',' ')

def seed_milvus(URI_link: str, collection_name: str, filename: str, directory: str) -> Milvus:
    """
    Hàm crawl dữ liệu trực tiếp từ URL và tạo vector embeddings trong Milvus
    Args:
        URL (str): URL của trang web cần crawl dữ liệu
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection trong Milvus
        doc_name (str): Tên định danh cho tài liệu được crawl
    """
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    local_data, doc_name = load_data_from_local(filename,directory)

    # print(local_data)
    documents = [
        Document(
            page_content=doc.get('page_content') or '',
            metadata= {
                'source': doc['metadata'].get('source') or '',
                'content_type': doc['metadata'].get('content_type') or 'text/plain',
                'title': doc['metadata'].get('title') or '',
                'description': doc['metadata'].get('description') or '',
                'language': doc['metadata'].get('language') or 'en',
                'doc_name': doc_name,
                'start_index': doc['metadata'].get('start_index') or 0
            }
        )
        for doc in local_data
    ]
    print('documents: ', documents)

    uuids = [str(uuid4()) for _ in range(len(documents))]



    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri":URI_link},
        collection_name=collection_name,
        drop_old=False
    )

        
    batch_size = 5
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        bactch_ids = uuids[i:i+batch_size]
        vectorstore.add_documents(documents=batch_docs, ids = bactch_ids)
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
        - Sử dụng model 'text-embedding-3-large' cho việc tạo embeddings khi truy vấn
    """
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,   
    )
    return vectorstore

def main():
    """
    Hàm chính để kiểm thử các chức năng của module
    Thực hiện:
        1. Test seed_milvus với dữ liệu từ file local 'stack.json'
    Chú ý:
        - Đảm bảo Milvus server đang chạy tại localhost:19530
        - Các biến môi trường cần thiết (như OPENAI_API_KEY) đã được cấu hình
    """
    # Test seed_milvus với dữ liệu local
    seed_milvus('http://localhost:19530', 'data_test', 'stack.json', 'data')

if __name__ == "__main__" : 
    main()
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from vncorenlp import VnCoreNLP
import re
import json
from langchain_community.document_loaders import RecursiveUrlLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

# Khởi tạo VnCoreNLP (tải trước model từ https://github.com/vncorenlp/VnCoreNLP)
annotator = VnCoreNLP("VnCoreNLP-1.2.jar", annotators="wseg", max_heap_size='-Xmx2g') 

# Khởi tạo embedding model cho tiếng Việt
vi_embeddings = HuggingFaceEmbeddings(
    model_name="keepitreal/vietnamese-sbert",
    model_kwargs={'device': 'cpu'}
)

def clean_text_vn(text):
    """Làm sạch văn bản tiếng Việt"""
    text = re.sub(r'\s+', ' ', text)  # Xóa khoảng trắng thừa
    text = re.sub(r'\[.*?\]', '', text)  # Xóa nội dung trong []
    return text.strip()

def semantic_vn_splitter(text, max_chunk_size=3000):
    """Tách văn bản tiếng Việt dựa trên ngữ nghĩa"""
    # Tách câu và từ
    sentences = []
    annotated_text = annotator.annotate(text)
    for sentence in annotated_text['sentences']:
        sentences.append(' '.join([word['form'] for word in sentence]))
    
    # Tạo các đoạn văn có nghĩa
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= max_chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def craw_web_base_loader(url_data):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    loader = WebBaseLoader(url_data, header_template=headers)
    docs = loader.load()
    print('Initial length:', len(docs))

    # Làm sạch và xử lý văn bản
    cleaned_docs = []
    for doc in docs:
        cleaned_content = clean_text_vn(doc.page_content)
        cleaned_docs.append(doc.__class__(page_content=cleaned_content, metadata=doc.metadata))
    
    # Sử dụng semantic splitter cho tiếng Việt
    text_splitter = SemanticChunker(
        embeddings=vi_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        add_start_index=True
    )
    
    # Tách văn bản theo ngữ nghĩa
    all_splits = []
    for doc in cleaned_docs:
        chunks = semantic_vn_splitter(doc.page_content)
        for chunk in chunks:
            all_splits.append(doc.__class__(page_content=chunk, metadata=doc.metadata))
    
    print('Final chunks:', len(all_splits))
    return all_splits

def save_data_locally(documents, filename, directory):
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)  # Tạo đường dẫn đầy đủ

    data_to_save = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents]  # Chuyển đổi documents thành định dạng có thể serialize
    
    # Lưu vào file JSON
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data_to_save, file, indent=4, ensure_ascii=False)
    print(f'Data saved to {file_path}')  # In thông báo lưu thành công

def main():
    # Crawl dữ liệu 
    url = 'https://xaydungchinhsach.chinhphu.vn/toan-van-nghi-dinh-168-2024-nd-cp-quy-dinh-xu-phat-vi-pham-hanh-chinh-ve-trat-tu-atgt-duong-bo-119241231164556785.htm'

    data = craw_web_base_loader(url)
    # Lưu dữ liệu vào thư mục data_v2
    save_data_locally(data, 'stack.json', 'data')
    print('data: ', data)  # In dữ liệu đã crawl

# Kiểm tra nếu file được chạy trực tiếp
if __name__ == "__main__":
    main()
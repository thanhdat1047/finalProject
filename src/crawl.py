import os
import re 
import json
from langchain_community.document_loaders import RecursiveUrlLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from dotenv import load_dotenv

#load_dotenv()
def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html,"html.parser")    # Phan tich cu phap html
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()      # Xoa khoang trang va trong thua

def craw_web_base_loader(url_data):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    loader = WebBaseLoader(url_data, header_template=headers)  # Tạo loader cơ bản
    docs = loader.load()  # Tải nội dung
    print('length: ', len(docs))  # In số lượng tài liệu
    
    # Chia nhỏ văn bản tương tự như trên
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
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

    # url ='https://chatai.com/7-must-try-prompts-for-every-business-owner-or-entrepreneur/'
    data = craw_web_base_loader(url)
    # Lưu dữ liệu vào thư mục data_v2
    save_data_locally(data, 'stack.json', 'data')
    print('data: ', data)  # In dữ liệu đã crawl

# Kiểm tra nếu file được chạy trực tiếp
if __name__ == "__main__":
    main()
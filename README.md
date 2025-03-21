## 1. Install library
   # Tải thư viện 
    - pip install -r requirements.txt

## 2. Setup and run Milvus Database
   - Khởi động Docker Desktop
   - Mở Terminal/Command Prompt, chạy lệnh:  
        docker compose up --build

## 3. Cài attu để xem db
  - docker run -p 8000:3000 -e MILVUS_URL=192.168.2.6:19530 zilliz/attu:v2.4
  # Note: Lấy ip local
  - Kiểm tra data đã aào Milvus chưa bằng cách truy cập: http://localhost:8000/#/databases/default/colletions

## 4. Run 
  # Crawl data
  python crawl.py

  # Seed data
  python seed_data.py

  # Run Attu 
  docker run -p 8000:3000 -e MILVUS_URL=192.168.1.7:19530 zilliz/attu:v2.4

  # View collection
  http://localhost:8000/#/databases/default/colletions

  # Run app
  streamlit run main.py

  # git









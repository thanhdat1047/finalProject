import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Mean Pooling - Tính trung bình có trọng số dựa trên attention mask
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Phần tử đầu tiên của model_output chứa tất cả token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def search_similar_contexts(query, data_with_embeddings, tokenizer, model, top_k=5):
    """
    Tìm kiếm các context tương tự nhất với câu query
    
    Args:
        query: Câu truy vấn
        data_with_embeddings: Dữ liệu với embeddings
        tokenizer: Tokenizer đã được khởi tạo
        model: Model đã được khởi tạo
        top_k: Số lượng kết quả trả về
        
    Returns:
        Danh sách các kết quả tương tự nhất
    """
    max_length = 256
    
    # Encode query
    encoded_query = tokenizer([query], padding=True, truncation=True, return_tensors='pt', max_length=max_length)
    
    with torch.no_grad():
        query_model_output = model(**encoded_query)
    
    # Tính embedding cho query
    query_embedding = mean_pooling(query_model_output, encoded_query['attention_mask']).numpy()
    
    # Lấy tất cả các embeddings từ dữ liệu
    content_embeddings = np.array([item['embedding'] for item in data_with_embeddings])
    
    # Tính độ tương đồng cosine
    similarity_scores = cosine_similarity(query_embedding, content_embeddings)[0]
    
    # Lấy top_k kết quả tương tự nhất
    top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        item = data_with_embeddings[idx]
        result = {
            'score': float(similarity_scores[idx]),
            'id': item.get('id', ''),
            'content': item.get('content', ''),
            'reference_id': item.get('reference_id', '')
        }
        
        # Thêm các thông tin hữu ích khác nếu có
        for field in ['chuong', 'muc', 'dieu', 'khoan']:
            if field in item:
                result[field] = item[field]
                
        results.append(result)
    
    return results

def main():
    # Tải mô hình và tokenizer
    print("Đang tải mô hình và tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
    model = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
    
    # Đường dẫn đến file JSON chứa embeddings
    embeddings_file_path = '../metadata/metadata_with_embeddings.json'
    
    # Đọc dữ liệu từ file JSON
    print(f"Đang đọc dữ liệu từ {embeddings_file_path}...")
    try:
        with open(embeddings_file_path, 'r', encoding='utf-8') as file:
            data_with_embeddings = json.load(file)
        print(f"Đã đọc {len(data_with_embeddings)} mục từ file embeddings")
    except Exception as e:
        print(f"Lỗi khi đọc file: {str(e)}")
        return
    
    # Chế độ tương tác
    while True:
        query =  input("\nNhập câu truy vấn (hoặc nhập 'exit' để thoát): ")
        if query.lower() == 'exit':
            break
        
        top_k =  3 #int(input("Số lượng kết quả muốn hiển thị: ") or "5")
        
        # Tìm kiếm
        print("\nĐang tìm kiếm các context tương tự...")
        results = search_similar_contexts(query, data_with_embeddings, tokenizer, model, top_k=top_k)
        
        # Hiển thị kết quả
        print(f"\n=== Top {len(results)} kết quả tương tự với câu query: '{query}' ===\n")
        for i, result in enumerate(results):
            print(f"Kết quả #{i+1} - Điểm tương đồng: {result['score']:.4f}")
            print(f"ID: {result['id']}")
            if 'reference_id' in result:
                print(f"Reference ID: {result['reference_id']}")
            if 'dieu' in result:
                print(f"Điều: {result['dieu']}")
            if 'khoan' in result:
                print(f"Khoản: {result['khoan']}")
            print(f"Nội dung: {result['content']}")
            print("-" * 100)

if __name__ == "__main__":
    main()
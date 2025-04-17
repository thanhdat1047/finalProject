import json
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import re
import os
from tqdm import tqdm

# Mean Pooling - Tính trung bình có trọng số dựa trên attention mask
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Phần tử đầu tiên của model_output chứa tất cả token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Hàm chia nội dung thành các phần nhỏ dựa trên số token thực tế
def split_content_by_tokens(content, tokenizer, max_tokens_per_chunk=230):
    """Chia nội dung thành các chunk dựa trên số token thực tế."""
    # Tokenize nội dung để kiểm tra số lượng token
    encoded = tokenizer.encode_plus(content, add_special_tokens=True)
    tokens = encoded['input_ids']
    
    # Nếu số token không vượt quá giới hạn, giữ nguyên nội dung
    if len(tokens) <= max_tokens_per_chunk:
        return [content]
    
    # Chia thành các câu
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', content)
    
    chunks = []
    current_chunk = ""
    current_token_count = 0
    
    for sentence in sentences:
        # Kiểm tra số token của câu hiện tại
        sentence_tokens = len(tokenizer.encode_plus(sentence, add_special_tokens=False)['input_ids'])
        
        # Nếu câu hiện tại + current_chunk vượt quá giới hạn và current_chunk không rỗng
        if current_token_count + sentence_tokens > max_tokens_per_chunk and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_token_count = sentence_tokens
        # Nếu câu hiện tại vượt quá giới hạn và không thể chia nhỏ hơn
        elif sentence_tokens > max_tokens_per_chunk:
            # Nếu có nội dung trong current_chunk, thêm vào chunks
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_token_count = 0
            
            # Chia câu dài thành các đoạn nhỏ hơn dựa trên số từ
            words = sentence.split()
            temp_chunk = ""
            temp_token_count = 0
            
            for word in words:
                word_with_space = word + " "
                word_tokens = len(tokenizer.encode_plus(word_with_space, add_special_tokens=False)['input_ids'])
                
                if temp_token_count + word_tokens <= max_tokens_per_chunk:
                    temp_chunk += word_with_space
                    temp_token_count += word_tokens
                else:
                    chunks.append(temp_chunk.strip())
                    temp_chunk = word + " "
                    temp_token_count = word_tokens
            
            if temp_chunk:
                current_chunk = temp_chunk
                current_token_count = temp_token_count
        else:
            current_chunk += " " + sentence
            current_token_count += sentence_tokens
    
    # Thêm chunk cuối cùng nếu có
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def main():
    # Tải mô hình và tokenizer
    print("Đang tải mô hình và tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
    model = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
    
    # Đường dẫn đến file JSON
    input_file_path = '../metadata/metadata.json'
    
    # Đọc dữ liệu từ file JSON
    print(f"Đọc dữ liệu từ {input_file_path}...")
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Kiểm tra nếu data không phải là list
        if not isinstance(data, list):
            for key in data:
                if isinstance(data[key], list):
                    data = data[key]
                    break
            if not isinstance(data, list):
                data = [data]
        
        print(f"Đã đọc {len(data)} mục từ file JSON")
    except Exception as e:
        print(f"Lỗi khi đọc file: {str(e)}")
        return
    
    # Chuẩn bị danh sách content để encode
    all_contents = []
    content_mapping = []  # Để theo dõi mối quan hệ giữa content và item gốc
    
    max_length = 256  # Độ dài tối đa cho mỗi token sequence
    
    print("Đang phân tích và chia nhỏ nội dung...")
    for item_index, item in enumerate(tqdm(data)):
        if 'content' not in item or not item['content']:
            continue
            
        content = item['content']
        if not isinstance(content, str):
            continue
            
        # Kiểm tra độ dài token của content và chia nhỏ nếu cần
        chunks = split_content_by_tokens(content, tokenizer, max_tokens_per_chunk=230)  # Dự phòng cho special tokens
        
        # Nếu có nhiều hơn 1 chunk
        if len(chunks) > 1:
            for i, chunk in enumerate(chunks):
                all_contents.append(chunk)
                content_mapping.append({
                    'item_index': item_index,
                    'chunk_index': i,
                    'is_chunked': True,
                    'total_chunks': len(chunks)
                })
        else:
            all_contents.append(content)
            content_mapping.append({
                'item_index': item_index,
                'chunk_index': 0,
                'is_chunked': False,
                'total_chunks': 1
            })
    
    # Tạo batch để xử lý hiệu quả hơn
    batch_size = 16
    total_batches = (len(all_contents) + batch_size - 1) // batch_size
    
    all_embeddings = []
    
    print("Đang tạo embeddings...")
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(all_contents))
        batch_contents = all_contents[start_idx:end_idx]
        
        print(f"Đang xử lý batch {i+1}/{total_batches}...")
        
        # Encode batch contents
        encoded_inputs = tokenizer(batch_contents, padding=True, truncation=True, 
                                   return_tensors='pt', max_length=max_length)
        
        with torch.no_grad():
            model_outputs = model(**encoded_inputs)
        
        # Lấy embeddings
        batch_embeddings = mean_pooling(model_outputs, encoded_inputs['attention_mask'])
        
        # Chuyển tensor thành list và thêm vào kết quả
        for embedding in batch_embeddings.numpy():
            all_embeddings.append(embedding.tolist())
    
    print("Đang tạo JSON với embedding...")
    
    # Tổ chức lại dữ liệu: Kết hợp các chunk thuộc cùng một item
    processed_items = {}
    
    for original_idx, mapping in enumerate(content_mapping):
        item_index = mapping['item_index']
        original_item = data[item_index]
        content = all_contents[original_idx]
        embedding = all_embeddings[original_idx]
        
        # Nếu item chưa được xử lý, tạo một bản sao
        if item_index not in processed_items:
            processed_items[item_index] = original_item.copy()
            processed_items[item_index]['chunks'] = []
            # Nếu item có nhiều chunk, tạo trường để lưu combined embedding sau này
            if mapping['total_chunks'] > 1:
                processed_items[item_index]['has_multiple_chunks'] = True
            else:
                processed_items[item_index]['has_multiple_chunks'] = False
                processed_items[item_index]['embedding'] = embedding
        
        # Thêm thông tin chunk vào item
        chunk_info = {
            'chunk_index': mapping['chunk_index'],
            'content': content,
            'embedding': embedding
        }
        processed_items[item_index]['chunks'].append(chunk_info)
    
    # Kết hợp embeddings của các chunks thành một embedding duy nhất cho mỗi item có nhiều chunk
    for item_index, item in processed_items.items():
        if item['has_multiple_chunks']:
            # Sắp xếp các chunks theo thứ tự đúng
            sorted_chunks = sorted(item['chunks'], key=lambda x: x['chunk_index'])
            
            # Kết hợp embeddings (lấy trung bình)
            combined_embedding = np.mean([np.array(chunk['embedding']) for chunk in sorted_chunks], axis=0).tolist()
            item['embedding'] = combined_embedding
    
    # Chuyển từ dict sang list
    result = list(processed_items.values())
    
    # Lưu kết quả vào file JSON mới
    output_file_path = os.path.splitext(input_file_path)[0] + "_with_embeddings.json"
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    
    print(f"Hoàn thành! Đã lưu dữ liệu với {len(result)} mục vào file {output_file_path}")
    print(f"Tổng số chunks đã xử lý: {len(all_contents)}")

if __name__ == "__main__":
    main()
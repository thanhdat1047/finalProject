import os
import json
import re

def extract_numbers(text):
    # Trích xuất số từ chuỗi như "chương 1" -> 1
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    return None

def process_chunk_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    
    # Trích xuất thông tin
    chuong = ""
    muc = ""
    dieu = ""
    khoan = ""
    
    for line in lines:
        if "chương" in line.lower() and not chuong:
            chuong = line.strip()
        elif "mục" in line.lower() and not muc:
            muc = line.strip()
        elif "điều" in line.lower() and not dieu:
            dieu = line.strip()
        elif "khoản" in line.lower() and not khoan:
            khoan = line.strip()
            # Phần còn lại là nội dung của khoản
            start_idx = lines.index(line)
            full_content = '\n'.join(lines[start_idx:])
            break
    
    # Tạo reference_id
    chuong_num = extract_numbers(chuong) if chuong else 0
    muc_num = extract_numbers(muc) if muc else 0
    dieu_num = extract_numbers(dieu) if dieu else 0
    khoan_num = extract_numbers(khoan) if khoan else 0
    
    reference_id = f"{chuong_num}.{muc_num}.{dieu_num}.{khoan_num}"
    
    # Tạo ID duy nhất từ tên file
    unique_id = os.path.splitext(os.path.basename(file_path))[0]
    
    # Tạo metadata
    metadata = {
        "id": unique_id,
        "chuong": chuong,
        "muc": muc,
        "dieu": dieu,
        "khoan": khoan,
        "content": full_content,
        "reference_id": reference_id
    }
    
    return metadata

def process_all_chunk_files(directory):
    metadata_list = []
    
    for filename in os.listdir(directory):
        if filename.startswith('chunk_') and filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            metadata = process_chunk_file(file_path)
            metadata_list.append(metadata)
            
            # Lưu metadata vào file JSON tương ứng
            json_filename = os.path.splitext(filename)[0] + '.json'
            json_path = os.path.join(directory, 'metadata', json_filename)
            
            # Đảm bảo thư mục metadata tồn tại
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Lưu tất cả metadata vào một file
    all_metadata_path = os.path.join(directory, 'metadata', 'all_metadata.json')
    with open(all_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    
    return metadata_list

# Sử dụng hàm
chunks_directory = "../output_4"  # Thay đổi thành đường dẫn thực tế
metadata = process_all_chunk_files(chunks_directory)
print(f"Đã tạo metadata cho {len(metadata)} file chunk.")
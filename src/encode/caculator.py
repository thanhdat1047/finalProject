import json

def find_longest_content(json_array):
    # Khởi tạo biến để lưu thông tin của phần tử có content dài nhất
    max_length = 0
    max_content_item = None
    
    # Duyệt qua từng phần tử trong mảng
    for item in json_array:
        # Kiểm tra nếu phần tử có trường content
        if "content" in item:
            # Lấy độ dài của content
            content_length = len(item["content"])
            
            # So sánh với độ dài lớn nhất hiện tại
            if content_length > max_length:
                max_length = content_length
                max_content_item = item
    
    # Trả về kết quả
    return {
        "max_length": max_length,
        "item": max_content_item
    }

# Đường dẫn đến file JSON
file_path = "../src/metadata/metadata.json"  # Thay đổi đường dẫn này

try:
    # Đọc dữ liệu từ file JSON
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    
    # Kiểm tra xem dữ liệu có phải là list không
    if not isinstance(json_data, list):
        # Nếu không phải list mà là dictionary có chứa list
        # Thử tìm kiếm trong các trường phổ biến
        for key in json_data:
            if isinstance(json_data[key], list):
                json_data = json_data[key]
                break
        # Nếu vẫn không phải list, chuyển đổi thành list với một phần tử
        if not isinstance(json_data, list):
            json_data = [json_data]
    
    # Gọi hàm để tìm phần tử có content dài nhất
    result = find_longest_content(json_data)
    
    # In kết quả
    print(f"Độ dài lớn nhất: {result['max_length']}")
    print(f"ID của phần tử có content dài nhất: {result['item']['id']}")
    print(f"Content dài nhất: {result['item']['content'][:100]}...")  # Chỉ hiển thị 100 ký tự đầu tiên
    
except FileNotFoundError:
    print(f"File không tồn tại: {file_path}")
except json.JSONDecodeError:
    print("File không phải định dạng JSON hợp lệ")
except Exception as e:
    print(f"Lỗi: {str(e)}")
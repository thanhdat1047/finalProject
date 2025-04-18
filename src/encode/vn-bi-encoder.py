from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings, we could use pyvi, underthesea, RDRSegment to segment words
# sentences = ['Cô ấy là một người vui_tính .', 'Cô ấy cười nói suốt cả ngày .']
sentences = ["khoản 7. phạt tiền từ 12.000.000 đồng đến 14.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:\na) điều khiển xe chạy quá tốc độ quy định trên 35 km/h;\nb) điều khiển xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có gắn động cơ đi vào đường cao tốc;\nc) dừng xe, đỗ xe trên đường cao tốc không đúng nơi quy định; không có báo hiệu bằng đèn khẩn cấp khi gặp sự cố kỹ thuật hoặc bất khả kháng khác buộc phải dừng xe, đỗ xe ở làn dừng xe khẩn cấp trên đường cao tốc; không có báo hiệu bằng đèn khẩn cấp, không đặt biển cảnh báo \"chú ý xe đỗ\" (hoặc đèn cảnh báo) về phía sau xe khoảng cách tối thiểu 150 mét khi dừng xe, đỗ xe trong trường hợp gặp sự cố kỹ thuật hoặc bất khả kháng khác buộc phải dừng xe, đỗ xe trên một phần làn đường xe chạy trên đường cao tốc."]
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
model = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings.shape)

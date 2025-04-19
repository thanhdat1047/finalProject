import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Mean Pooling - Tính trung bình có trọng số dựa trên attention mask
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Phần tử đầu tiên của model_output chứa tất cả token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embedding_query(query: str, tokenizer, model ):
    max_length = 256
    
    # Encode query
    encoded_query = tokenizer([query], padding=True, truncation=True, return_tensors='pt', max_length=max_length)
        
    with torch.no_grad():
        query_model_output = model(**encoded_query)
    
    # Tính embedding cho query
    query_embedding = mean_pooling(query_model_output, encoded_query['attention_mask']).numpy()
    
    return query_embedding
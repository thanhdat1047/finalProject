import os 
import json 
import torch
import numpy as np
from typing import List ,Dict, Any, Optional, Union

from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_core.tools import ToolException

from transformers import AutoTokenizer, AutoModel
from astrapy import DataAPIClient, Database, Collection
from dotenv import load_dotenv

from embeddings import  vi_embedding

load_dotenv()

class VectorSearchInput(BaseModel):
    query: str = Field(...,description="Cau hoi hoac truy van cua nguoi dung")
    top_k: int = Field(default=5, description="So luong ket qua tra ve")
    
class VectorSearchTool(BaseTool):
    name: str = "vector_search"
    description: str = "Tìm kiếm thông tin tương tự dựa trên câu truy vấn"
    args_schema: type[BaseModel] = VectorSearchInput
    
    def __init__(
        self,
        collection_name: str = "law_vi",
        tokenizer_name: str = "bkai-foundation-models/vietnamese-bi-encoder",
        model_name: str = "bkai-foundation-models/vietnamese-bi-encoder",
        db_endpoint: Optional[str] = None,
        db_token: Optional[str] = None
    ):
        super().__init__()
        
        # Load tokenizer and model
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._collection_name = collection_name
        
        # Setup DB
        self._db_endpoint = db_endpoint or os.getenv("ASTRA_DB_API_ENDPOINT")
        self._db_token = db_token or os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        
        
        if not self._db_endpoint or not self._db_token:
            raise ValueError(
                "Database endpoint và token phải được cung cấp qua tham số hoặc biến môi trường"
            )
        
        # Connect to DB
        self._database = self._connect_to_database()
        self._collection = self._connect_to_collection()
        
    def _connect_to_database(self) -> Database:
        try:
            client = DataAPIClient()
            database = client.get_database(self._db_endpoint,token=self._db_token)
            return database
        except Exception as e:
            raise ToolException(f"Không thể kết nối đến database: {str(e)}")
        
    def _connect_to_collection(self) -> Collection:
        try:
            collection = self._database.get_collection(self._collection_name)
            return collection
        except Exception as e:
            raise ToolException(f"Không thể kết nối đến collection {self._collection_name}: {str(e)}")
        
    def _generate_embedding(self, query: str) -> List[float]:
        try:
            embedding = vi_embedding.embedding_query(query, self._tokenizer, self._model)
            return embedding.flatten().tolist()
        except Exception as e:
            raise ToolException(f"Lỗi khi tạo embedding: {str(e)}")
        
    def _search_similar_vectors(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        try:
            results = self._collection.find(
                {},
                sort={"$vector": query_embedding},
                limit=top_k,
                include_similarity=True,
            )
            return list(results)
        except Exception as e:
            raise ToolException(f"Lỗi khi tìm kiếm vector tương tự: {str(e)}")
        
    def _run(self, query: str, top_k: int = 5, run_manager: Optional[CallbackManagerForToolRun] = None) -> Union[str, List[Dict]]:
        try:
            # create query_embedding
            query_embedding = self._generate_embedding(query)
            
            # search_similar
            results = self._search_similar_vectors(query_embedding, top_k)
            
            return results
        except Exception as e:
            raise ToolException(f"Lỗi khi thực hiện tìm kiếm: {str(e)}")        
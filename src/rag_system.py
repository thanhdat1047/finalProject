import os 
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent

from retrieve import vector_search_tool

load_dotenv()

class LawRAG: 
    def __init__(self):
        self.llm = GoogleGenerativeAI(
            temperature=0,
            model="gemini-1.5-flash-001", # Đảm bảo có quyền truy cập model này
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        self.vector_search_tool = vector_search_tool.VectorSearchTool(collection_name="law_vi")
        
        self._setup_rag_pipeline()

    def _setup_rag_pipeline(self):
        
        self.retrieval_prompt = ChatPromptTemplate.from_template(
            """You are a professional Vietnamese legal assistant. Based on the information below, please answer the user's question accurately and completely.

            Reference Information:
            {context}

            Question: {query}

            If the information is not sufficient to answer, please say that you do not have enough information to answer this question."""
        )
        
        self.rag_chain = (
            {"context": self._retrieve_documents, "query": RunnablePassthrough()}
            | self.retrieval_prompt
            | self.llm
            | StrOutputParser()
        )
        
        self._setup_agent()
    
    def _retrieve_documents(self, query: str) -> str:
        results = self.vector_search_tool._run(query=query, top_k=5)
        
        if not results:
            return "Không tìm thấy thông tin liên quan."
        
        context_text = []
        for result in results:
            similarity = result.get('$similarity', 0)
            content = result.get('content', '')
            chuong = result.get('chuong', '')
            muc = result.get('muc', '')
            dieu = result.get('dieu', '')
        
            if similarity > 0.7:
                context_text.append(f"{chuong}\n{muc}\n{dieu}\n{content}")
        
        return "\n\n".join(context_text) if context_text else "Không tìm thấy thông tin đủ liên quan."
    
    def _setup_agent(self):
        tools = [self.vector_search_tool]
        
        prompt = ChatPromptTemplate.from_template(
            """Bạn là một trợ lý pháp luật Việt Nam chuyên nghiệp.
            Sử dụng các công cụ được cung cấp để trả lời câu hỏi của người dùng.
            
            {chat_history}
            
            Công cụ có sẵn:
            {tools}
        
            Sử dụng các công cụ sau:
            {tool_names}
            
            Câu hỏi của người dùng: {input}
            {agent_scratchpad}
            """
        )
        
        self.agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent= self.agent,
            tools= tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
    def query(self, query: str) -> str:
        return self.rag_chain.invoke(query)
    
    def chat(self, query: str, chat_history: List = None) -> str:
        if chat_history is None:
            chat_history = []
        
        response = self.agent_executor.invoke(
            {"input":query, "chat_history": chat_history}
        )
        
        return response["output"]

if __name__ == "__main__":
    # Khởi tạo hệ thống RAG
    rag_system = LawRAG()
    
    # Ví dụ sử dụng
    query = "Không đội mũ bảo hiểm khi lái xe máy bị phạt bao nhiêu tiền?"
    print(f"Câu hỏi: {query}")
    
    # Sử dụng RAG đơn giản
    result = rag_system.query(query)
    print(f"Trả lời (RAG): {result}")
    
    # # Sử dụng agent
    # agent_result = rag_system.chat(query)
    # print(f"Trả lời (Agent): {agent_result}")
import os
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.tools.render import render_text_description
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from pydantic import BaseModel
import streamlit as st

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")


class SearchInput(BaseModel):
    query: str

def get_retriever(collection_name: str = "legal_data") -> EnsembleRetriever:  # Đổi tên collection cho phù hợp
    """
    Tạo một ensemble retriever kết hợp vector search (Milvus) và BM25
    Args:
        collection_name (str): Tên collection trong Milvus để truy vấn
    """
    try:
        # Connect to Milvus
        from seed import connect_to_milvus  # Import ở đây để tránh lỗi nếu không seed data
        vectorstore = connect_to_milvus('http://localhost:19530',
                                        collection_name)
        milvus_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # Tạo BM25 retriever từ toàn bộ documents
        documents = [
            Document(page_content=doc.page_content,
                    metadata=doc.metadata)
            for doc in vectorstore.similarity_search("", k=100)
        ]

        if not documents:
            raise ValueError(f"Không tìm thấy documents trong collection '{collection_name}'")

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4

        # Kết hợp hai retriever với tỷ trọng
        ensemble_retriever = EnsembleRetriever(
            retrievers=[milvus_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        return ensemble_retriever
    except Exception as e:
        print(f"Lỗi khi khởi tạo retriever: {str(e)}")
        # Trả về retriever với document mặc định nếu có lỗi
        default_doc = [
            Document(
                page_content="Có lỗi xảy ra khi kết nối database. Vui lòng thử lại sau.",
                metadata={"source": "error"}
            )
        ]
        return BM25Retriever.from_documents(default_doc)

# Tao cong cu tim kiem cho agent
#"""Tìm kiếm thông tin từ retriever"""

def create_search_tool(retriever):
    def search_tool(query: str)-> str:
        try:
            # Làm sạch query
            clean_query = " ".join(query.split()[:10])
            
            docs = retriever.invoke(clean_query,  config={"k": 5})
            if not docs:
                return "Không tìm thấy dữ liệu phù hợp"
                
            return "\n\n".join([f"[Doc {i+1}]: {d.page_content}" for i, d in enumerate(docs)])
        except Exception as e:
            return f"Lỗi hệ thống: {str(e)}"

    return StructuredTool(
        name="find_luat_giao_thong",
        func=search_tool,
        description="Tra cứu luật giao thông. Dùng khi cần thông tin về: phạt, điều luật, nghị định",
        args_schema=SearchInput
    )

retriever = get_retriever()
search_tool = create_search_tool(retriever)

def format_scratchpad(steps):
    return "\n".join([
        f"Thought: {action.log}\nObservation: {observation}"
        for action, observation in steps
    ])

def get_llm_and_agent(_retriever, model_choice="gemini") -> AgentExecutor:
    """
    Khởi tạo Language Model và Agent với cấu hình cụ thể
    Args:
        _retriever: Retriever đã được cấu hình để tìm kiếm thông tin
        model_choice: Lựa chọn model ("gpt4" hoặc "gemini")
    """

    llm = GoogleGenerativeAI(
        temperature=0,
        model="gemini-1.5-flash-001", # Đảm bảo có quyền truy cập model này
        api_key=GOOGLE_API_KEY
    )

    general_tool = Tool.from_function(
        name="general_questions",
        func=lambda q: "Tôi chỉ trả lời về luật giao thông Việt Nam",
        description="Dùng cho các câu hỏi không liên quan đến luật giao thông"
    )

    tools = [
        general_tool,
        create_search_tool(_retriever)
    ]

    system_template = """You MUST always respond in the following format:
    **Thought**: [Analyze the question and plan how to answer]
    **Action**: [Tool name if needed, e.g., find_luat_giao_thong]
    **Action Input**: [Input for the tool]
    **Observation**: [Result from the tool]
    **Final Answer**: [Final response in Vietnamese]

    Rules:
    1. For traffic law-related questions, use find_luat_giao_thong
    2. If tool results are UNRELATED to the question, immediately stop and respond: 'Không tìm thấy quy định' (No regulations found)
    3. Only use tools when legal information is needed. MAX 3 tool calls
    4. For non-traffic law questions, use general_questions
    5. Synthesize information from multiple results instead of verbatim responses
    6. If results exceed 1000 characters, request user to provide more details
    7. FINAL ANSWER MUST BE IN VIETNAMESE

    Important Notes:
    - Maintain professional tone for legal responses
    - Cross-check information between different documents
    - If conflicting information is found, state: 'Có sự khác biệt trong quy định. Vui lòng tham khảo Nghị định số...' 
    - Always conclude with **Final Answer** in Vietnamese"""
    

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        MessagesPlaceholder("chat_history", optional=True),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])

    # initialize_agent
    agent = initialize_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,

        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True),

        output_parser=ReActSingleInputOutputParser(),

        handle_parsing_errors=lambda _: "Lỗi định dạng! Vui lòng trả lời ngắn gọn." ,
        format_scratchpad=format_scratchpad,

        max_iterations=3,  # Giới hạn số lần gọi tool
        early_stopping_method="generate" 
    )

    return agent

agent_executor = get_llm_and_agent(retriever)

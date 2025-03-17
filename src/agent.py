from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import Tool,AgentExecutor, create_openai_functions_agent, create_react_agent , create_tool_calling_agent # Tao & thuc thi agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate # Xu ly prompt
from seed_data import seed_milvus, connect_to_milvus 
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory # Luu tru lich su chat
from langchain.retrievers import EnsembleRetriever # Ket hop nhieu retriever
from langchain_community.retrievers import BM25Retriever 
from langchain_core.documents import Document
from langchain.tools.render import render_text_description
from dotenv import load_dotenv
import os

load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

def get_retriever(collection_name: str = "data_test") -> EnsembleRetriever:
    """
    Tạo một ensemble retriever kết hợp vector search (Milvus) và BM25
    Args:
        collection_name (str): Tên collection trong Milvus để truy vấn
    """  
    try:
        # Connect to Milvus 
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
tool = create_retriever_tool(
    get_retriever(),
    "find",
    "Search for information of Stack AI."
)

def get_llm_and_agent(_retriever, model_choice="gpt4") -> AgentExecutor:
    """
    Khởi tạo Language Model và Agent với cấu hình cụ thể
    Args:
        _retriever: Retriever đã được cấu hình để tìm kiếm thông tin
        model_choice: Lựa chọn model ("gpt4" hoặc khac)
    """
    # llm = ChatOpenAI(
    #     temperature=0,
    #     streaming=True,
    #     model='gpt-3.5-turbo',
    #     api_key=OPENAI_API_KEY
    # )

    llm = GoogleGenerativeAI(
        temperature=0,
        model="gemini-1.5-pro",
        api_key=GOOGLE_API_KEY
    )
    tools =[tool]

    system = """You are an expert at AI. Your name is ChatchatAI."""
    prompt = ChatPromptTemplate([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    

    # prompt = prompt.partial(
    #     tools=render_text_description(tools),
    #     tool_names=", ".join([t.name for t in tools]),
    # )

    # agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Khoi tao retrieve and agent 
# retriever = get_retriever()
# agent_executor = get_llm_and_agent(retriever)
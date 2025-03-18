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
import streamlit as st

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

def get_retriever(collection_name: str = "data_test") -> EnsembleRetriever:  # Đổi tên collection cho phù hợp
    """
    Tạo một ensemble retriever kết hợp vector search (Milvus) và BM25
    Args:
        collection_name (str): Tên collection trong Milvus để truy vấn
    """
    try:
        # Connect to Milvus
        from seed_data import connect_to_milvus  # Import ở đây để tránh lỗi nếu không seed data
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
def create_search_tool(retriever):
    def search_tool(query: str):
        """Tìm kiếm thông tin từ retriever."""
        return retriever.get_relevant_documents(query)

    return Tool(
        name="find_luat_giao_thong",
        func=search_tool,
        description="Tìm kiếm thông tin liên quan đến luật giao thông Việt Nam. Sử dụng công cụ này để tra cứu các điều luật, quy định, mức phạt và các thông tin pháp lý khác về giao thông."
    )

retriever = get_retriever()
search_tool = create_search_tool(retriever)
tools = [search_tool]

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

    tools = [create_search_tool(_retriever)]

    system = """Bạn là một chuyên gia tư vấn về luật giao thông Việt Nam. Bạn có kiến thức sâu rộng về các quy định, điều luật, và mức phạt liên quan đến giao thông. Hãy sử dụng công cụ 'find_luat_giao_thong' để tra cứu thông tin cần thiết và trả lời câu hỏi của người dùng một cách chính xác và đầy đủ. Nếu không tìm thấy thông tin, hãy nói rõ là bạn không thể trả lời câu hỏi."""  # Mô tả rõ ràng vai trò và cách sử dụng tool

    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", system),
    #     MessagesPlaceholder(variable_name="chat_history"),
    #     ("user", "{input}"),
    #     ("assistant", "{agent_scratchpad}"),
    # ])

    prompt = PromptTemplate(
        system = system,
        template=
        """
        {system}

        {chat_history}

        Question: {input}

        {agent_scratchpad}
        """
    )
    
    system_template = """Bạn là một AI về lĩnh vực luật giao thông, bạn chuyên tư vấn về luật giao thông Việt Nam.Không trả lời các câu hỏi thuộc lĩnh vực khác luật giao thông .Bạn có kiến thức sâu rộng về các quy định, điều luật, và mức phạt liên quan đến giao thông. Hãy sử dụng công cụ 'find_luat_giao_thong' để tra cứu thông tin cần thiết và trả lời câu hỏi của người dùng một cách chính xác và đầy đủ. Nếu không tìm thấy thông tin, hãy nói rõ là bạn không thể trả lời câu hỏi."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "{input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    ai_template = "{agent_scratchpad}"
    ai_message_prompt = AIMessagePromptTemplate.from_template(ai_template)

    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        MessagesPlaceholder(variable_name="chat_history"),
        human_message_prompt,
        ai_message_prompt
    ])


    # Initialize memory
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    #Sử dụng create_react_agent thay vì initialize_agent
    agent = initialize_agent(
        llm=llm,
        tools=tools,
        prompt=chat_prompt,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent
    # agent = initialize_agent(
    #     llm=llm, tools=tools, prompt=prompt
    # )

    # return AgentExecutor(agent=agent, tools=tools, verbose=True)


# # Khoi tao retrieve and agent
# retriever = get_retriever()
agent_executor = get_llm_and_agent(retriever)

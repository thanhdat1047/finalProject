"""
File chính để chạy ứng dụng Chatbot AI
Chức năng: 
- Tạo giao diện web với Streamlit
- Xử lý tương tác chat với người dùng
- Kết nối với AI model để trả lời
"""

# === IMPORT thu vien
import streamlit as st
from dotenv import load_dotenv
from seed_data import seed_milvus
from agent import get_retriever , get_llm_and_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# === Thiet lap giao dien
def setup_page():
    """
    Cau hinh trang web co ban
    """
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="💬",
        layout="wide" # Giao dien rong 
    )
# === Khoi tao ung dung
def initalize_app():
    """
    Khoi tao cac cai dat can thiet
    - Doc file env
    - Cau hinh trang web
    """
    load_dotenv()
    setup_page()

# === Thanh cong cu

def setup_sidebar():
    """
    Tao thanh cong cu ben trai voi cac  tuy chon 
    """
    with st.sidebar:
        st.title("⚙️ Cấu hình")

        # Phan 1: Emmbedding model
        st.header("🔤 Embeddings Model")
        st.write("GPT") 
        # use_openAI_embeddings = ("GPT") 

        # Phan 2: Cau hinh Data
        st.header("📚 Nguồn dữ liệu")
        st.write("File local") 

        handle_local_file() # Seed du lieu

        # Them phan chon collection de query
        st.header("🔍 Collection để truy vấn")
        collection_to_query = st.text_input(
            "Nhap ten collection",
            "data_test",
            help="Nhap ten collection su dung de tim kiem thong tin"
        )

        # Phan 3: Chon Model de tra loi
        st.header("🤖 Model AI")
        model_choice = "OpenAI GPT-4"
        st.write(model_choice)

        return model_choice, collection_to_query

def handle_local_file():
    collection_name = st.text_input(
        "Ten collection:",
        "data_test",
        help="Nhap ten collection muon luu vao Milvus"
    )
    filename = st.text_input("Ten file JSON:", "stack.json")
    directory = st.text_input("Thuc muc chua file:", "data")
    
    if st.button("Tai du lieu tu file"): 
        if not collection_name:
            st.error("Vui long nhap ten collection!")
            return
        
        with st.spinner("Dang tai du lieu ..."):
            try:
                seed_milvus(
                    'http://localhost:19530',
                    collection_name,
                    filename,
                    directory
                )
                st.success(f"Da tai du lieu thanh cong vao collection '{collection_name}'!")
            except Exception as e:
                st.error(f"Loi khi tai du lieu: {str(e)}")


# === Giao dien chat chinh ===
def setup_chat_interface():
    st.title("💬 AI Assistant")

    st.caption("🚀 Trợ lý AI được hỗ trợ bởi LangChain và OpenAI GPT-4")

    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}
        ]

        msgs.add_ai_message("Toi co the giup gi cho ban?")

    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs

# === Xu ly tin nhan nguoi dung 
def handle_user_input(msgs, agent_executor):
    """
    Xu ly khi nguoi dung gui tin nhan 
    1. Hien thi tin nhan nguoi dung 
    2. Goi AI xu ly va tra loi
    3. Luu vao lich su chat
    """
    if prompt := st.chat_input("Hãy hỏi tôi bất cứ điều gì về luật giao thông đường bộ"):
        # Luu va hien thi tin nhan nguoi dung
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        msgs.add_user_message(prompt)

        # Xu ly va hien thi cau tra loi 
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())

            # Lay lich su chat
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]

            # Goi AI xu ly
            response = agent_executor.invoke(
                {
                    "input": prompt,
                    "chat_history": chat_history
                },
                {
                    "callbacks": [st_callback]
                }
            )

            # Luu va hien thi cau tra loi 
            output = response["output"]
            st.session_state.messages.append({"role": "assistant", "content": output})
            msgs.add_ai_message(output)
            st.write(output)

# === Ham chinh ===
def main(): 
    """
    Ham chinh dieu khien luong chuong trinh
    """
    initalize_app()
    model_choice,collection_to_query = setup_sidebar()
    msgs = setup_chat_interface()

    retriever = get_retriever(collection_to_query)
    agent_executor = get_llm_and_agent(retriever, "gpt4")

    handle_user_input(msgs, agent_executor)

# Run
if __name__ == "__main__":
    main()




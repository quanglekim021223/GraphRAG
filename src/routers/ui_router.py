import streamlit as st
import os
import uuid
import asyncio
import nest_asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from src.helpers.logging_config import logger
from src.config.settings import Config
from src.helpers.agent_initializer import initialize_agent
from src.handlers.conversation_handler import store_conversation, get_conversation_history, get_all_conversations


def main():
    """Main Streamlit application."""
    # Enable nested event loops for asyncio
    nest_asyncio.apply()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    st.title("Healthcare Assistant with GraphRAG")
    st.write(
        "Hỏi tôi bất cứ điều gì về sức khỏe hoặc bệnh nhân! (Hỗ trợ tiếng Việt và tiếng Anh)")

    # Custom CSS for the chat input
    st.markdown(
        """
        <style>
        .stChatInput {
            position: sticky;
            bottom: 0;
            z-index: 100;
            background-color: #1e1e2f;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    try:
        required_vars = ["NEO4J_URI", "NEO4J_USERNAME",
                         "NEO4J_PASSWORD", "GITHUB_TOKEN"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            st.error(
                f"Missing environment variables: {', '.join(missing_vars)}")
            return

        # Initialize agent
        agent_executor = initialize_agent()

        # Initialize session state
        if "conversations" not in st.session_state:
            st.session_state.conversations = get_all_conversations()
        if "current_thread_id" not in st.session_state:
            st.session_state.current_thread_id = None
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Chat", "Memory", "Conversations"])

        # Tab Chat
        with tab1:
            # Conversation selector
            st.subheader("Chọn hoặc tạo cuộc hội thoại")
            conversation_options = st.session_state.conversations + \
                ["Tạo cuộc hội thoại mới"]
            selected_conversation = st.selectbox(
                "Chọn cuộc hội thoại:", conversation_options)

            if selected_conversation == "Tạo cuộc hội thoại mới":
                new_thread_id = str(uuid.uuid4())
                st.session_state.conversations.append(new_thread_id)
                st.session_state.current_thread_id = new_thread_id
                st.session_state.messages = []
            else:
                if st.session_state.current_thread_id != selected_conversation:
                    st.session_state.current_thread_id = selected_conversation
                    st.session_state.messages = get_conversation_history(
                        selected_conversation)

            # Display conversation history
            st.subheader("Lịch sử hội thoại")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            # Add scroll button
            if st.button("Cuộn xuống để nhập câu hỏi"):
                st.rerun()

            # Chat input
            user_input = st.chat_input("Nhập câu hỏi của bạn...")
            if user_input:
                st.session_state.messages.append(
                    {"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.write(user_input)

                try:
                    system_message = SystemMessage(content="""
                    You are a healthcare assistant. Based on the user's question:
                    - If the question is about specific patient data, diseases, doctor, hospital, insurance provider, room or treatments, use the 'rag_tool'.
                    - If the question is general or no specific data is needed, use the 'llm_tool'.
                    """)

                    config = {"configurable": {
                        "thread_id": st.session_state.current_thread_id}}
                    response = agent_executor.invoke(
                        {"messages": [system_message,
                                      HumanMessage(content=user_input)]},
                        config
                    )["messages"][-1].content

                    st.session_state.messages.append(
                        {"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.write(response)

                    # Save to Neo4j
                    store_conversation(
                        st.session_state.current_thread_id, user_input, response)
                except Exception as e:
                    logger.error(
                        f"Error processing question: {str(e)}", exc_info=True)
                    st.error(
                        f"Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}. Vui lòng thử lại.")

            # Delete conversation button
            if st.button("Xóa lịch sử cuộc hội thoại hiện tại"):
                try:
                    config = Config()
                    from src.handlers.graphrag_handler import HealthcareGraphRAG
                    graphrag = HealthcareGraphRAG(config)
                    with graphrag.graph_manager.graph._driver.session() as session:
                        session.run(
                            """
                            MATCH (c:Conversation {thread_id: $thread_id})-[:HAS_MESSAGE]->(m:Message)
                            DETACH DELETE c, m
                            """,
                            {"thread_id": st.session_state.current_thread_id}
                        )
                    st.session_state.conversations.remove(
                        st.session_state.current_thread_id)
                    st.session_state.current_thread_id = None
                    st.session_state.messages = []
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error deleting conversation: {str(e)}")
                    st.error(f"Đã xảy ra lỗi khi xóa lịch sử: {str(e)}.")

        # Tab Memory
        with tab2:
            st.subheader("Memory Preview (based on current conversation)")
            if st.session_state.messages:
                st.write("Memory content for current conversation:")
                for msg in st.session_state.messages:
                    st.write(f"{msg['role'].capitalize()}: {msg['content']}")
            else:
                st.write(
                    "No memory content available in the current conversation.")

        # Tab Conversations
        with tab3:
            st.subheader("Danh sách các cuộc hội thoại")
            if st.session_state.conversations:
                for thread_id in st.session_state.conversations:
                    st.write(f"Cuộc hội thoại: {thread_id}")
                    history = get_conversation_history(thread_id)
                    if history:
                        for msg in history:
                            st.write(
                                f"{msg['role'].capitalize()}: {msg['content']}")
                    else:
                        st.write("Không có lịch sử cho cuộc hội thoại này.")
                    st.markdown("---")
            else:
                st.write("Chưa có cuộc hội thoại nào.")

    except ValueError as e:
        logger.error(f"Startup failed: {str(e)}")
        st.error(f"Error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        st.error(f"Error: {str(e)}")


def run_ui():
    """Run the Streamlit UI."""
    main()


if __name__ == "__main__":
    run_ui()

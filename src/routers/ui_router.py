"""
UI Router module for Healthcare GraphRAG system.

This module provides a Streamlit-based web interface for the Healthcare GraphRAG system,
enabling interactive chat with memory management, conversation history visualization,
and topic extraction from healthcare conversations.
"""
import os
import uuid
import asyncio

import nest_asyncio
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from src.helpers.agent_initializer import agent_initializer
from src.helpers.logging_config import logger
from src.helpers.prompts import get_healthcare_system_prompt
from src.handlers.conversation_handler import (
    store_conversation,
    get_conversation_history,
    get_all_conversations,
    delete_conversation
)


def generate_conversation_name(messages):
    """
    Create a readable name for a conversation based on first user message.

    Args:
        messages: List of message dictionaries with role and content

    Returns:
        String containing a human-readable conversation name
    """
    if not messages:
        return "Cuộc hội thoại mới"

    # Lấy tin nhắn đầu tiên của user
    for message in messages:
        if message["role"] == "user":
            content = message["content"]
            # Cắt nội dung để tạo tên ngắn gọn
            words = content.split()[:4]  # Lấy tối đa 4 từ đầu
            name = " ".join(words)
            if len(name) > 30:
                name = name[:27] + "..."
            return name

    return "Cuộc hội thoại mới"


def render_conversation_selector():
    """
    Render the conversation selector UI component.

    Returns:
        Boolean: True if UI state changed and rerun is needed
    """
    st.subheader("Chọn hoặc tạo cuộc hội thoại")
    conversation_options = []
    conversation_map = {}  # Map UUID to friendly name

    # Generate friendly names for all conversations
    for thread_id in st.session_state.conversations:
        history = get_conversation_history(thread_id)
        friendly_name = generate_conversation_name(history)
        # Thêm UUID ở cuối để đảm bảo không trùng lặp
        display_name = f"{friendly_name} ({thread_id[:6]})"
        conversation_options.append(display_name)
        conversation_map[display_name] = thread_id

    # Add option to create new conversation
    conversation_options.append("Tạo cuộc hội thoại mới")

    selected_option = st.selectbox(
        "Chọn cuộc hội thoại:", conversation_options)

    if selected_option == "Tạo cuộc hội thoại mới":
        new_thread_id = str(uuid.uuid4())
        st.session_state.conversations.append(new_thread_id)
        st.session_state.current_thread_id = new_thread_id
        st.session_state.messages = []
        # Update URL with new thread_id
        st.query_params.update(thread_id=new_thread_id)
        return True
    else:
        selected_thread_id = conversation_map[selected_option]
        if st.session_state.current_thread_id != selected_thread_id:
            st.session_state.current_thread_id = selected_thread_id
            st.session_state.messages = get_conversation_history(
                selected_thread_id)
            # Update URL with selected thread_id
            st.query_params.update(thread_id=selected_thread_id)
            return True

    return False


def process_user_input(agent_executor, user_input):
    """
    Process user input and generate response using the agent.

    Args:
        agent_executor: The LangGraph agent instance
        user_input: User's question string
    """
    if not user_input:
        return

    st.session_state.messages.append(
        {"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    try:
        with st.spinner('Đang xử lý câu hỏi của bạn...'):
            # Lấy context đầy đủ từ lịch sử hội thoại
            conversation_context = agent_initializer.get_conversation_context(
                st.session_state.current_thread_id)

            # Sử dụng prompt chung
            system_content = get_healthcare_system_prompt(conversation_context)
            system_message = SystemMessage(content=system_content)

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
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Exceptions need to be caught broadly as this is a top-level UI handler
        logger.error("Error processing question: %s", str(e), exc_info=True)
        st.error(
            f"Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}. Vui lòng thử lại.")


def render_memory_tab():
    """Render the Memory tab displaying conversation context and topics."""
    st.subheader("Bộ nhớ hội thoại")
    if st.session_state.current_thread_id:
        memory = agent_initializer.get_memory(
            st.session_state.current_thread_id)
        context = memory.get_conversation_context()

        st.markdown("### Lịch sử hội thoại")
        st.text_area("Nội dung hội thoại", value=context.get("conversation", ""),
                     height=300, disabled=True)

        st.markdown("### Chủ đề chính")
        if "topics" in context and context["topics"]:
            for topic in context["topics"]:
                st.markdown(f"- {topic}")
        else:
            st.write("Chưa có chủ đề nào được nhận diện.")
    else:
        st.info(
            "Vui lòng chọn hoặc tạo một cuộc hội thoại để xem thông tin bộ nhớ.")


def render_conversations_tab():
    """Render the Conversations tab displaying all conversation history."""
    st.subheader("Danh sách các cuộc hội thoại")
    if st.session_state.conversations:
        for thread_id in st.session_state.conversations:
            history = get_conversation_history(thread_id)
            friendly_name = generate_conversation_name(history)

            # Tạo container có viền và padding
            with st.container():
                st.markdown(f"### {friendly_name}")
                st.caption(f"ID: {thread_id}")

                if history:
                    # Chỉ hiển thị 2-3 tin nhắn đầu
                    for msg_idx, msg in enumerate(history[:3]):
                        st.write(
                            f"{msg['role'].capitalize()}: {msg['content'][:50]}...")
                    if len(history) > 3:
                        st.caption(
                            f"... và {len(history)-3} tin nhắn khác")
                else:
                    st.write(
                        "Không có lịch sử cho cuộc hội thoại này.")
                st.markdown("---")
    else:
        st.write("Chưa có cuộc hội thoại nào.")


def main():
    """Main Streamlit application."""
    # Cấu hình động cho tiêu đề trang dựa trên thread_id hiện tại
    thread_id_display = "New Chat"
    query_params = st.query_params
    if "thread_id" in query_params:
        thread_id = query_params["thread_id"]
        if "current_thread_id" not in st.session_state or st.session_state.current_thread_id != thread_id:
            st.session_state.current_thread_id = thread_id
            st.session_state.messages = get_conversation_history(thread_id)

    if "current_thread_id" in st.session_state and st.session_state.current_thread_id:
        thread_id_display = f"Thread: {st.session_state.current_thread_id[:8]}..."

    st.set_page_config(
        page_title=f"Healthcare GraphRAG - {thread_id_display}",
        page_icon="🏥",
        layout="wide"
    )
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
        agent_executor = agent_initializer.get_agent()

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
            # Render conversation selector and check if rerun is needed
            if render_conversation_selector():
                st.rerun()

            # Add scroll button
            if st.button("Cuộn xuống để nhập câu hỏi"):
                st.rerun()

            # Chat input
            user_input = st.chat_input("Nhập câu hỏi của bạn...")
            process_user_input(agent_executor, user_input)

            # Delete conversation button
            if st.button("Xóa lịch sử cuộc hội thoại hiện tại"):
                try:
                    if delete_conversation(st.session_state.current_thread_id):
                        # Nếu xóa thành công, cập nhật UI
                        st.session_state.conversations.remove(
                            st.session_state.current_thread_id)
                        st.session_state.current_thread_id = None
                        st.session_state.messages = []
                        st.rerun()
                    else:
                        st.error(
                            "Không thể xóa cuộc hội thoại. Xem log để biết thêm chi tiết.")
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error(
                        "Error when deleting conversation: %s", str(e))
                    st.error(f"Đã xảy ra lỗi khi xóa lịch sử: {str(e)}.")

        # Tab Memory
        with tab2:
            render_memory_tab()

        # Tab Conversations
        with tab3:
            render_conversations_tab()

    except ValueError as e:
        logger.error("Startup failed: %s", str(e))
        st.error(f"Error: {str(e)}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        # This is the top-level error handler
        logger.error("Unexpected error: %s", str(e), exc_info=True)
        st.error(f"Error: {str(e)}")


def run_ui():
    """Run the Streamlit UI."""
    main()


if __name__ == "__main__":
    run_ui()

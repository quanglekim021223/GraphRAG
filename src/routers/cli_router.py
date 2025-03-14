from src.config.settings import Config
from src.helpers.logging_config import logger
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.helpers.agent_initializer import initialize_agent
import uuid


def run_cli():
    """Run the application in CLI mode."""
    config = Config()
    config.validate()

    # Initialize ReAct agent with both tools
    agent_executor = initialize_agent()

    print("Healthcare GraphRAG CLI")
    print("Type 'exit' to quit")

    # Generate a permanent thread_id for this CLI session
    session_thread_id = str(uuid.uuid4())
    print(f"Session ID: {session_thread_id}")

    while True:
        question = input("\nEnter your question: ")
        if question.lower() in ['exit', 'quit', 'q']:
            break

        try:
            # Use only the agent approach with both tools
            system_message = SystemMessage(content="""
                You are a healthcare assistant. Based on the user's question:
                - If the question is about specific patient data, diseases, doctor, hospital, insurance provider, room or treatments, use the 'rag_tool'.
                - If the question is general or no specific data is needed, use the 'llm_tool'.
            """)

            # Provide thread_id in configurable
            config_obj = {"configurable": {"thread_id": session_thread_id}}

            # Invoke agent and get full response
            full_response = agent_executor.invoke(
                {"messages": [system_message, HumanMessage(content=question)]},
                config_obj
            )

            # Extract the response content
            agent_response = full_response["messages"][-1].content
            print(f"\n🔍 Response: {agent_response}")

            # Try to display query info if available in the agent logs
            if "logs" in full_response:
                import re
                log_text = full_response.get("logs", "")
                match = re.search(r"MATCH\s+.*RETURN.*",
                                  log_text, re.IGNORECASE | re.DOTALL)
                if match:
                    query_info = match.group(0)
                    print(f"\n📊 Cypher Query: {query_info}")

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    run_cli()

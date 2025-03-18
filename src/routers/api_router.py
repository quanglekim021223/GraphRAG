from flask import Flask, request, jsonify
from src.config.settings import Config
from src.helpers.logging_config import logger
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.helpers.agent_initializer import agent_initializer
import uuid
import re


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Load configuration
    config = Config()
    config.validate()

    # Initialize ReAct agent with both tools
    agent_executor = agent_initializer.get_agent()

    @app.route('/', methods=['GET'])
    def home():
        return "Welcome to the Healthcare GraphRAG chatbot!"

    @app.route('/chat', methods=['POST'])
    def chat():
        data = request.json
        question = data.get('question')
        # Use provided thread_id or generate new one
        thread_id = data.get('thread_id', str(uuid.uuid4()))

        if not question:
            return jsonify({"error": "Question is required"}), 400

        try:
            # Use only the agent approach
            system_message = SystemMessage(content="""
                You are a healthcare assistant. Based on the user's question:
                - If the question is about specific patient data, diseases, doctor, hospital, insurance provider, room or treatments, use the 'rag_tool'.
                - If the question is general or no specific data is needed, use the 'llm_tool'.
            """)

            # Provide thread_id in configurable
            config_obj = {"configurable": {"thread_id": thread_id}}

            # Invoke agent and get full response
            full_response = agent_executor.invoke(
                {"messages": [system_message, HumanMessage(content=question)]},
                config_obj
            )

            # Extract the response content
            agent_response = full_response["messages"][-1].content

            # Try to extract query information from the agent logs
            query_info = None
            if "logs" in full_response:
                # Look for Cypher query in the logs
                log_text = full_response.get("logs", "")
                match = re.search(r"MATCH\s+.*RETURN.*",
                                  log_text, re.IGNORECASE | re.DOTALL)
                if match:
                    query_info = match.group(0)

            return jsonify({
                "question": question,
                "response": agent_response,
                "query": query_info,
                "thread_id": thread_id  # Return the thread_id for future requests
            })
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            return jsonify({"error": str(e)}), 500

    return app


def run_api(port=5000):
    """Run the Flask application."""
    app = create_app()
    app.run(host='0.0.0.0', port=port)


if __name__ == "__main__":
    run_api()

#!/bin/bash

set -e

# Đảm bảo URI đúng trong mọi trường hợp
if [[ "$NEO4J_URI" == *"localhost"* ]]; then
  echo "Incorrect NEO4J_URI detected. Fixing it..."
  export NEO4J_URI="bolt://neo4j:7687"
fi

echo "Using Neo4j URI: $NEO4J_URI"

if command -v neo4j-admin &> /dev/null; then
    echo "Dumping Neo4j database..."
    neo4j-admin dump --database=neo4j --to=/backups/neo4j.dump
    echo "Dump completed."
else
    echo "Neo4j admin tool not available - skipping dump"
fi


# Đợi Neo4j khởi động
echo "Waiting for Neo4j to be ready..."
echo "Current Neo4j URI: $NEO4J_URI" 
for i in {1..30}; do
    if nc -z neo4j 7687; then
        echo "Neo4j is up!"
        break
    fi
    echo "Waiting for Neo4j... $i/30"
    sleep 2
done

# Kiểm tra nếu vẫn không kết nối được sau 30 lần thử
if ! nc -z neo4j 7687; then
    echo "Error: Could not connect to Neo4j after 30 attempts"
    echo "Checking Neo4j connectivity details:"
    echo "Neo4j URI: $NEO4J_URI"
    echo "Neo4j Username: $NEO4J_USERNAME"
    exit 1
fi

# Thêm đoạn kiểm tra kết nối chi tiết bằng Python
echo "Testing Neo4j connection with Python..."
python -c "
import os
from neo4j import GraphDatabase
uri = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
username = os.getenv('NEO4J_USERNAME', 'neo4j')
password = os.getenv('NEO4J_PASSWORD', 'healthcaregraphrag')
print(f'Connecting to: {uri} with user {username}')
try:
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        with driver.session() as session:
            result = session.run('RETURN 1 AS result').single().value()
            print(f'Success! Result: {result}')
except Exception as e:
    print(f'Error: {str(e)}')
"

# Khởi động ứng dụng dựa trên MODE
if [ "$MODE" = "api" ]; then
    echo "Starting API mode on port $PORT"
    python main.py --mode api --port $PORT
elif [ "$MODE" = "ui" ]; then
    echo "Starting UI mode on port 8501"
    # Chạy trực tiếp Streamlit thay vì thông qua Python
    streamlit run src/routers/ui_router.py
elif [ "$MODE" = "cli" ]; then
    echo "Starting CLI mode"
    python main.py --mode cli
else
    echo "Invalid MODE: $MODE. Must be one of: api, ui, cli"
    exit 1
fi
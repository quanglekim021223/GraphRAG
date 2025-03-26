#!/bin/bash

# Đảm bảo quyền thực thi
set -e

echo "Starting Neo4j entrypoint script..."

# Kiểm tra xem có file dump trong thư mục backups không
if [ -f "/backups/neo4j.dump" ]; then
  echo "Found neo4j.dump file. Preparing to import..."

  # Đợi Neo4j khởi động đủ để sử dụng các lệnh admin
  until cypher-shell -u $NEO4J_USERNAME -p $NEO4J_PASSWORD "RETURN 1;" > /dev/null 2>&1; do
    echo "Waiting for Neo4j to be ready before import..."
    sleep 5
  done

  # Dừng database để import
  echo "Stopping Neo4j database for import..."
  cypher-shell -u $NEO4J_USERNAME -p $NEO4J_PASSWORD "STOP DATABASE neo4j" || true

  # Import dump
  echo "Importing database from dump file..."
  neo4j-admin database load --verbose --overwrite-destination=true --from-path=/backups neo4j

  # Khởi động lại database
  echo "Starting Neo4j database after import..."
  cypher-shell -u $NEO4J_USERNAME -p $NEO4J_PASSWORD "START DATABASE neo4j"

  echo "Database import completed!"
  
  # Kiểm tra schema và cập nhật metadata
  echo "Updating schema metadata..."
  cypher-shell -u $NEO4J_USERNAME -p $NEO4J_PASSWORD "CALL db.schema.visualization();"
  cypher-shell -u $NEO4J_USERNAME -p $NEO4J_PASSWORD "CALL apoc.meta.schema();"
fi

# Chạy Neo4j với chế độ console để giữ container chạy
echo "Starting Neo4j server in console mode..."
exec neo4j console
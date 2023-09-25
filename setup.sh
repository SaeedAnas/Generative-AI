# Write to db
python SemanticSearch/src/helpers/db.py

# Create mappings
curl -XPUT -H "Content-Type: application/json" -d @config/mappings/es-mapping.json http://localhost:9200/topic_text.public.chunks

# Create connectors
curl -i -X POST -H "Accept:application/json" -H "Content-Type:application/json" 127.0.0.1:8083/connectors/ --data "@config/connectors/es-sink.json"
curl -i -X POST -H "Accept:application/json" -H "Content-Type:application/json" 127.0.0.1:8083/connectors/ --data "@config/connectors/postgres-text.json"
curl -i -X POST -H "Accept:application/json" -H "Content-Type:application/json" 127.0.0.1:8083/connectors/ --data "@config/connectors/postgres-vector.json"
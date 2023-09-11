#!/bin/bash

# location of tika server jar, edit this
export TIKA_SERVER_JAR="file:///Users/praveen/dev/project-SV/Assignment1/tika-jar/tika-server-standard-2.9.0.jar"

# location of input document file
echo "Input document files:  /Users/praveen/dev/project-SV/Assignment1/data_files"

# Database configuration, 
export DB_HOST="localhost"
export DB_NAME="postgres"
export DB_PORT="5432"
# Edit this
export DB_USER="praveen"
export DB_PASSWORD="password"

# Elastic search, edit
export ELASTIC_URL="https://localhost:9200"
export ELASTIC_PASSWORD="2f7_-fUYsNvSoPtaa*be"
export ELASTIC_CERT_PATH="~/dev/database/kibana-8.9.2/data/ca_1694198172681.crt"

# Drop tables
#psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "DROP TABLE IF EXISTS documents CASCADE;"
#psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "DROP TABLE IF EXISTS chunks CASCADE;"
#psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "DROP TABLE IF EXISTS metadata CASCADE;"

echo "Tables 'chunks' and 'metadata' and 'documents' dropped successfully!"

#python setup_db.py
#python SSearch11_store.py
python SSearch22_index.py
#python SSearch3_query.py
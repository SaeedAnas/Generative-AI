#!/bin/bash

# Source the configuration file
source config.txt

# Assuming the tika-jar directory is located in your Git repository directory
export TIKA_SERVER_JAR="file://$PWD/data/tika-jar/tika-server-standard-2.9.0.jar"

# Drop tables
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "DROP TABLE IF EXISTS documents CASCADE;"
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "DROP TABLE IF EXISTS chunks CASCADE;"
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "DROP TABLE IF EXISTS metadata CASCADE;"

echo "Tables 'chunks' and 'metadata' and 'documents' dropped successfully!"

python src/helpers/setup_db.py
python src/SSearch1_store.py
#python SSearch2_index.py
#python SSearch3_query.py

import pymysql
# Establish a connection

def run():
    conn = pymysql.connect(
        host="192.168.1.83",
        user="chandar",
        password="chandar",
        database="documents"
    )

    # Create a cursor
    cursor = conn.cursor()

    # Insert a row of data
    cursor.execute("INSERT INTO DOCUMENT (ID, UUID, FILE_NAME, TEXT, DOC_TYPE, LANGUAGE) VALUES (%s, %s, %s, %s, %s, %s)", (7, 'a', 'b', 'c', 'd', 'e'))

    # Commit the changes
    conn.commit()

    # Close the connection
    cursor.close()
    conn.close()

if __name__ == '__main__':
    run()

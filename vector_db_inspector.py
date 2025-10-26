import sqlite3
from pprint import pprint

# Path to your Chroma DB
db_path = "vector_stores/chroma.sqlite3"

# Connect
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Fetch collections
cursor.execute("SELECT id, name FROM collections")
collections = cursor.fetchall()

pprint(collections)

conn.close()

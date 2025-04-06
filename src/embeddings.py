import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase

load_dotenv()

driver = GraphDatabase.driver(
    uri=os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
)

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="text-embedding-ada-002"
)

# Function to store embedding in Neo4j


def store_embedding(tx, uuid, embedding):
    query = """
    MATCH (q:Question) WHERE q.uuid = $uuid AND q.embedding IS NULL
    SET q.embedding = $embedding
    """
    tx.run(query, uuid=uuid, embedding=embedding)

# Function to process each node and store embeddings


def process_nodes():
    with driver.session() as session:
        # Get all nodes with the 'title' property
        result = session.run(
            "MATCH (q:Question) WHERE q.title IS NOT NULL AND q.embedding IS NULL RETURN q.uuid AS uuid, q.title AS title"
        )

        record_count = len(list(result))
        print(f'Found {record_count} questions without embeddings')

        i = 0
        for record in result:
            i += 1
            print(f'\nProcessing {i} record...\n ')

            uuid = record["uuid"]
            title = record["title"]

            # Generate embedding for the title
            embedding = embedding_provider.embed_query(title)

            # Store embedding in Neo4j
            session.execute_write(store_embedding, uuid, embedding)


# Run the function to process nodes and store embeddings
process_nodes()

def create_vector_index():
    # Create the Neo4jGraph
    graph = Neo4jGraph(
        url=os.getenv('NEO4J_URI'),
        username=os.getenv('NEO4J_USERNAME'),
        password=os.getenv('NEO4J_PASSWORD')
    )

    # Create the vector index
    graph.query("""
        CREATE VECTOR INDEX `titleVector`
        IF NOT EXISTS
        FOR (q:Question) ON (q.embedding)
        OPTIONS {indexConfig: {
        `vector.similarity_function`: 'cosine'
        }};
    """)

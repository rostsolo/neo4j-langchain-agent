from llm import llm
from graph import graph

from langchain_neo4j import GraphCypherQAChain
from langchain.prompts import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """
Task:Generate Cypher statement to query a graph database.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Only include the generated Cypher statement in your response.

Always use case insensitive search when matching strings.

Example Cypher Statements:

1. To check how many questions on StackOverflow do not have an answer:
MATCH (q:Question) WHERE NOT ()-[:ANSWERS]->(q) RETURN count(q) AS result

2. To find the most popular topic on StackOverflow:
MATCH (q:Question)-[:TAGGED]->(t:Tag) RETURN t.name AS tag, count(q) AS count ORDER BY count DESC LIMIT 1

Schema:
{schema}

The question is:
{question}
"""

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True,
    allow_dangerous_requests=True
)
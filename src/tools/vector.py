from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from llm import llm
from embeddings import embedding_provider
from graph import graph
from langchain_core.prompts import ChatPromptTemplate


# Create the Neo4jVector
from langchain_neo4j import Neo4jVector

neo4jvector = Neo4jVector.from_existing_index(
    embedding_provider,
    graph=graph,
    index_name="titleVector",
    node_label="Question",
    text_node_property="title",
    embedding_node_property="embedding",
    retrieval_query="""
RETURN
    node.title AS text,
    score,
    {dummy: 'no_metadata'} AS metadata
"""
)

# Create the retriever
retriever = neo4jvector.as_retriever()

# Create prompt with instructions
instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

# Create chain to retrieve similar questions and their answers
question_answer_chain = create_stuff_documents_chain(llm, prompt)
answer_retriever = create_retrieval_chain(
    retriever,
    question_answer_chain
)


def get_answer_for_question(input):
    return answer_retriever.invoke({"input": input})

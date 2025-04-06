# Neo4j LangChain Agent

An agent designed to answer StackOverflow questions using OpenAI's LLM, LangChain, and the Neo4j graph database. This project is created for educational purposes.

---

## Table of Contents

1. [Overview](#overview)
2. [Reference](#reference)
3. [Setup Instructions](#setup-instructions)
    - [Install Dependencies](#install-dependencies)
    - [Configure Environment Variables](#configure-environment-variables)
    - [Prepare the Neo4j Database](#prepare-the-neo4j-database)
4. [Running the Application](#running-the-application)

---

## Overview

This project demonstrates how to integrate OpenAI's language models with Neo4j and LangChain to create a chatbot capable of answering StackOverflow questions. It leverages semantic search and graph-based data storage for enhanced query handling.

---

## Reference

The implementation in this repository is inspired by the online course: [Build a Neo4j-backed Chatbot using Python course](https://graphacademy.neo4j.com/courses/llm-chatbot-python/) available on [GraphAcademy](https://graphacademy.neo4j.com).

---

## Setup Instructions

### Install Dependencies

To install the required Python libraries, run the following command:

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Ensure the following environment variables are set in a `.env` file located in the root directory and in a `secrets.toml` file within the `.streamlit` directory:

```txt
# OpenAI Configuration
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4"

# Neo4j Configuration
NEO4J_URI = "bolt://"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = ""
```

### Prepare the Neo4j Database

1. Create a Neo4j database in the Sandbox using the StackOverflow template.
2. After the database is created, run the following script to generate embeddings for each question title:

    ```bash
    python src/embeddings.py
    ```

    This script will:
    - Generate embeddings for each question title.
    - Create a `titleVector` vector index to enable semantic search.

---

## Run the Application

Once the setup is complete, start the application using the following command:

```bash
streamlit run src/bot.py
```

The application will be accessible at [http://localhost:8501/](http://localhost:8501/).

---

Authored by Rostyslav Solopatych
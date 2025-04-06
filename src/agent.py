from utils import get_session_id
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.tools import Tool
from llm import llm
from graph import graph
from tools.cypher import cypher_qa
from tools.vector import get_answer_for_question

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser

# General chat
general_chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a StackOverflow portal expert providing answers on software engineering questions using questions and answers provided on the portal."),
        ("human", "{input}"),
    ]
)

general_chat = general_chat_prompt | llm | StrOutputParser()

# Create a set of tools
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat when all other tools do not have a good answer",
        func=general_chat.invoke,
    ),
    Tool.from_function(
        name="Software engineering assistant",
        description="For when you need to answer a question related to software engineering, programming languages or technology",
        func=get_answer_for_question
    ),
    Tool.from_function(
        name="StackOverflow assistant",
        description="For providing information about entities on StackOverflow portal like questions and answers using Cypher",
        func=cypher_qa
    )
]

# Create chat history callback


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


# Create the agent
agent_prompt = PromptTemplate.from_template("""
You are a StackOverflow platform expert providing information about software engineering.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to software engineering, programming languages or technology.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Create a handler to call the agent


def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st


# initialize the database connection
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)


def get_sql_chain(db):
    template = """You are a data analyst at a company. You are interacting with a user who is asking you questions 
    about the company's database. Based on the table schema below, write a SQL query that would answer the user's 
    question. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA> 
    Conversation History: {chat_history}
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: Count the total number of entries in the dataset.  
    SQL Query: SELECT COUNT(*) AS total_entries FROM suicides_data;
    
    Question: State-wise Analysis:Count the number of suicides in each state.
    SQL Query:SELECT State, COUNT(*) AS total_suicides FROM suicides_data GROUP BY State;
    
    Question: Year-wise Analysis:Count the number of suicides each year.
    SQL Query: SELECT Year, COUNT(*) AS total_suicides FROM suicides_data GROUP BY Year;
    
    Question: Type-wise Analysis:Count the number of suicides based on different types.
    SQL Query: SELECT Type, COUNT(*) AS total_suicides FROM suicides_data GROUP BY Type;
    
    Question: Gender-wise Analysis:Count the number of suicides based on gender.
    SQL Query: SELECT Gender, COUNT(*) AS total_suicides FROM suicides_data GROUP BY Gender;
    
    Question: Specific State and Year Analysis:Count the number of suicides in a specific state and year.
    SQL Query: SELECT State, Year, COUNT(*) AS total_suicides FROM suicides_data GROUP BY State, Year;
    
    Your turn:
    Question: {question}
    SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model="gpt-4-0125-preview")

    def get_schema(_):
        return db.get_table_info()
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)

    template = """You are a data analyst at a company. You are interacting with a user who is asking you questions 
    about the company's database. 
    Based on the table schema below, question, SQL query , write a natural language response.

    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    Question: {question}
    SQL Response: {response}"""
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model="gpt-4-0125-preview")
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),

    ]

load_dotenv()
st.set_page_config(page_title="Chat with your Database", layout="wide", page_icon="���")
st.title("Chat with your Database")

with st.sidebar:
    st.sidebar.title("Settings")
    st.write("This is a simple chat application to communicate with your Database")

    host = st.text_input("Host", value="localhost", key="Host")
    port = st.text_input("Port", value="3306", key="Port")
    user = st.text_input("User", value="root", key="User")
    password = st.text_input("Password", type="password", value="admin", key="Password")
    database = st.text_input("Database", value="sucides", key="Database")

    if st.button("Connect"):
        with st.spinner("Connecting to your Database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to your Database!!..")
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
user_query = st.chat_input("Type your message here..")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
        st.session_state.chat_history.append(AIMessage(content=response))

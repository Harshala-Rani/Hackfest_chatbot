from dotenv import load_dotenv
import os
from langchain.schema import (
    HumanMessage
)
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import streamlit as st
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection
import boto3
from langchain.chains import LLMChain
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
AWS_REGION = os.getenv("AWS_REGION")
OPENAPI_KEY = os.getenv("OPENAPI_KEY")
auth = (os.getenv("USERNAME"), os.getenv("PASSWORD"))
# Connect to OpenSearch
def connect_to_opensearch():
    host = 'search-opensearch-service-eemb5tvpntsoc2tu6s2jyvk2uq.us-west-1.es.amazonaws.com'
    region = AWS_REGION
    service = 'es'
    credentials = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN
    ).get_credentials()
    auth = AWSV4SignerAuth(credentials, region, service)
    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=30
    )
    print(client)
    if client.ping():
        print('Connected to OpenSearch')
        return client
    else:
        print("failed",client.info())
client = connect_to_opensearch()
# Streamlit interface
st.title("Chatbot Interface")
# User input
user_query = st.chat_input("Enter your query:")
if user_query:
    st.write(user_query)
    chat = ChatOpenAI(api_key=OPENAPI_KEY, model='gpt-4-0125-preview')
    # Change the user-input query to a text searchable in OpenSearch
    prompt_template = PromptTemplate(
        input_variables=["user_query"],
        template="You are an intelligent chatbot that can modify user queries to be more search-friendly. Extract the key words from the user query to make it more search-friendly: {user_query} For example if the user query is 'Explain the incident IR-308' extracted information should be 'incident IR-308'."
    )
    response = chat([HumanMessage(content=prompt_template.format(user_query=user_query))])
    modified_query = str(response.content)
    st.write(modified_query)
    # Search OpenSearch index
    response = client.search(
        index='aggreGenie',
        body={
            "_source": {
                "excludes": [
                    "text_vector"
                ]
            },
            "size": 5,
            "query": {
                "neural": {
                    "text_vector": {
                        "query_text": modified_query,
                        "model_id": "vsa7J5IBKdJdvEdqbLSP",
                        "k": 5
                    }
                }
            }
        }
    )
    # Extract relevant documents
    hits = response['hits']['hits']
    documents = [hit['_source']['text'] for hit in hits]
    # Create prompt template
    prompt_template = PromptTemplate(
        input_variables=["user_query", "documents"],
        template="User query: {user_query}\nRelevant documents: {documents}\nAnswer:"
    )
    # Generate response
    prompt = prompt_template.format(user_query=user_query, documents=documents)
    response = chat([HumanMessage(content=prompt)])
    # Display response
    st.write(response.content)
    # Create a chain
    chain = LLMChain(
        llm=chat,
        prompt=prompt_template
    )
    # Talk with the chain
    response = chain.run(user_query=user_query, documents=documents)
    # Display response
    st.write(response)
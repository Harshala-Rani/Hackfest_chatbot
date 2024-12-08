import json
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import os
import boto3
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import streamlit as st
from langchain.schema import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

load_dotenv()

# Load environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
AWS_REGION = os.getenv("AWS_REGION")
OPENAPI_KEY = os.getenv("OPENAI_API_KEY")

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
    else:
        print("failed",client.info())
    return client



def search_opensearch(client, index, query_text, model_id, size=5):
    """
    Perform a search in OpenSearch using a neural query.

    Args:
        client (OpenSearch): The OpenSearch client object.
        index (str): The index to search in.
        query_text (str): The query text to search for.
        model_id (str): The model ID used for vector search.
        size (int): The number of top results to return. Default is 5.

    Returns:
        list: A list of text content from the matching documents.
    """
    try:
        # Perform the search
        response = client.search(
            index=index,
            body={
                "_source": {"excludes": ["text_vector"]},
                "size": size,
                "query": {
                    "neural": {
                        "text_vector": {
                            "query_text": query_text,
                            "model_id": model_id,
                            "k": size
                        }
                    }
                }
            }
        )
        # Extract relevant documents
        hits = response['hits']['hits']
        documents = [hit['_source']['text'] for hit in hits]
        return documents
    except Exception as e:
        print(f"Error during OpenSearch query: {e}")
        return []


# Initialize Streamlit app
st.title("💬 Chatbot with OpenSearch")
st.caption("🚀 A Streamlit chatbot powered by OpenAI and OpenSearch")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
# Process user input
if prompt := st.chat_input():
    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Determine whether to use knowledge base or OpenAI
    chat = ChatOpenAI(api_key=OPENAPI_KEY, model="gpt-4-0125-preview")
    guidance_prompt_template = PromptTemplate(
        input_variables=["user_query"],
        template=(
            "You are a smart assistant. When given a user query, decide if it requires looking up information in the "
            "knowledge base or generating a direct response. "
            "Respond with 'search knowledge base' if you need to search or 'generate response' otherwise.\n"
            "User query: {user_query}"
        )
    )
    guidance_prompt = guidance_prompt_template.format(user_query=prompt)
    guidance_response = chat([HumanMessage(content=guidance_prompt)]).content.strip().lower()

    if "search knowledge base" in guidance_response:
        # Connect to OpenSearch and search knowledge base
        client = connect_to_opensearch()
        results = search_opensearch(client, prompt)
        client.transport.close()
        
        # Format and display search results
        if results:
            response_content = "Here are some results from the knowledge base:\n"
            for result in results:
                response_content += f"- {result['_source']['title']}: {result['_source']['content'][:100]}...\n"
        else:
            response_content = "No relevant documents found in the knowledge base."
    else:
        # Generate direct response using OpenAI
        response = chat([HumanMessage(content=prompt)])
        response_content = response.content

    # Append assistant's message to chat history and display
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    st.chat_message("assistant").write(response_content)
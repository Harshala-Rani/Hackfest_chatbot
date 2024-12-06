import json
from openai import OpenAI
import streamlit as st
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import os
import boto3
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
AWS_REGION = os.getenv("AWS_REGION")

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
    return client

# Generate query vector using a sentence transformer model
def get_query_vector(text):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Convert tensor to list of floats
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Perform a KNN search in OpenSearch
def search_opensearch(client, query_vector):
    print("query_vector",query_vector)
    query = {
        "size": 5,
        # "query": {
        #     "knn": {
        #         "field": "text_vector",
        #         "query_vector": query_vector,
        #         "k": 5
        #     }
        # }
        "query": {
   "script_score": {
     "query": {
       "match_all": {}
     },
     "script": {
       "source": "knn_score",
       "lang": "knn",
       "params": {
         "field": "text_vector",
         "query_value": [2.0, 3.0, 5.0, 6.0],
         "space_type": "cosinesimil"
         
       }
     }
   }
 }
    }
    try:
        response = client.search(body=query, index='confluence_kb')
    except Exception as e:
        print("Error:", e)
        return
    return response['hits']['hits']

# Initialize Streamlit app
st.title("💬 Chatbot with OpenSearch")
st.caption("🚀 A Streamlit chatbot powered by OpenAI and OpenSearch")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)
    
    # Ask the model if it should query the knowledge base
    guidance_prompt = f"Does the following request require information from a knowledge base? '{prompt}'"
    guidance_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": guidance_prompt}]
    )
    
    if "yes" in guidance_response.choices[0].message.content.lower():
        # Generate query vector
        query_vector = get_query_vector(prompt)
        
        # Perform the OpenSearch query
        client = connect_to_opensearch()
        results = search_opensearch(client, query_vector)
        client.transport.close()
        
        # Format and display search results
        if results:
            response_content = "Here are some results from the knowledge base:\n"
            for result in results:
                response_content += f"- {result['_source']['title']}: {result['_source']['content'][:100]}...\n"
        else:
            response_content = "No relevant documents found in the knowledge base."
        
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        st.chat_message("assistant").write(response_content)
    else:
        # Use OpenAI for general conversation
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages
        )
        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
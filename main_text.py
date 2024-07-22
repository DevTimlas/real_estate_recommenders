from flask import Flask, request, jsonify
import requests
import pandas as pd
from rich import print
from warnings import filterwarnings
from bs4 import BeautifulSoup
import emoji
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.preprocessing import normalize
import faiss

filterwarnings('ignore')

app = Flask(__name__)

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
import json

def get_q(msg):
    llm = ChatGroq(temperature=0, model="llama3-70b-8192",
                       api_key="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")
    memory = ConversationBufferMemory(memory_key="get_bed", return_messages=True)
    sys_prompt = f"""The user says: "{msg}", Just incase the text is long and not straightforward, Look for the main 
    information in the user's input and extract it. main information is the number of bedrooms, so, anything the user type
    just take note of the number of bedrooms and return something like 3 bedroom or 2 bedroom, and map it with bedroom
    
    also extract the location and map it with location, also extract if the house is for Rent or for Sale and map it to sub_type,
    if for rent just return Rent or for sale just return Sale
    then return the whole words and map it to query
    finally return a dictionary... only return the dict.. no suffix or prefix, no additional message! the dict is all we need
    and return just empty string if any of the three keywords aren't found, and return others that are found
    if any of the key parameter is passed, just automatically pass none and map it to the key that the value is not provided
    for example the user might not mention of they're looking for a house that's for rent or for sale.. in that case, just pass the
    unknown keyword to the sub_type, and if they didn't mention number of bedroom they want, just pass unknown keyword to bedroom
    likewise for location too
    """
    prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(sys_prompt),
                                               MessagesPlaceholder(variable_name="get_bed"),
                                               HumanMessagePromptTemplate.from_template(f"{msg}")])
    conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)

    memory.chat_memory.add_user_message(msg)
    response = conversation.invoke({"text": msg})
    # print("entity", response['text'])
    # print(json.loads(response['text']))
    return json.loads(response['text'])


# Function to fetch data from a specific page
def fetch_data(page):
    url = f'https://sapi.hauzisha.co.ke/api/properties/search?per_page=1000&page={page}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data for page {page}. Status code: {response.status_code}")
        return None

# Function to clean HTML tags from string
def clean_text(raw_html):
    clean_text = BeautifulSoup(raw_html, "html.parser").get_text()
    clean_text = clean_text.replace('📌', '').strip()
    clean_text = emoji.replace_emoji(clean_text, replace='')
    return clean_text

# Function to embed text
def embed_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to calculate Inner Product from L2 score
def calculateInnerProduct(L2_score):
    import math
    return (2 - math.pow(L2_score, 2)) / 2

# Function to search FAISS index
def searchFAISSIndex(data, query, index, nprobe, tokenizer, model, topk=20):
    description_embedding = embed_text(query['query'], tokenizer, model)
    location_embedding = embed_text(query['location'], tokenizer, model)
    bedroom_embedding = embed_text(query['bedroom'], tokenizer, model)
    submission_type_embedding = embed_text(query['sub_type'], tokenizer, model)
    
    query_embedding = np.hstack([description_embedding, location_embedding, bedroom_embedding, submission_type_embedding])
    query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    
    index.nprobe = nprobe
    D, I = index.search(query_embedding, topk)
    
    indices = I[0]
    L2_score = D[0]
    inner_product = [calculateInnerProduct(l2) for l2 in L2_score]
    
    matching_data = data.iloc[indices]
    
    search_result = pd.DataFrame({
        'index': indices,
        'cosine_sim': inner_product,
        'L2_score': L2_score
    })
    
    dat = pd.concat([matching_data.reset_index(drop=True), search_result.reset_index(drop=True)], axis=1)
    dat = dat.sort_values('cosine_sim', ascending=False)
    
    return dat

@app.route('/search', methods=['POST'])
def search():
    inp = request.json
    # print(inp)
    query = get_q(inp.get("query"))
    print(query)
    topk = inp.get('k', 20)  # Get 'k' from the request, default to 20 if not provided
    

    # Initialize an empty list to store data from all pages
    all_data = []

    # Fetch data from pages 1 to 5
    for page in range(1, 5):
        data = fetch_data(page)
        if data and 'data' in data:
            all_data.extend(data['data'])
        else:
            break

    # Convert the collected data into a Pandas DataFrame
    if all_data:
        properties = pd.json_normalize(all_data)
    else:
        return jsonify({"error": "No data was fetched from the API."})

    properties['description'] = properties['description'].str.lower()
    properties['description'] = properties['description'].apply(clean_text)
    properties = properties.dropna(axis=1, how='all')
    properties = properties.dropna(subset=['location', 'bedrooms', 'description'])

    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2', cache_dir='./cache_L')
    model = AutoModel.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2', cache_dir='./cache_L')

    properties['description_embedding'] = properties['description'].apply(lambda x: embed_text(x, tokenizer, model))
    properties['location_embedding'] = properties['location'].apply(lambda x: embed_text(x, tokenizer, model))
    properties['bedroom_embedding'] = properties['bedrooms'].apply(lambda x: embed_text(x, tokenizer, model))
    properties['submission_type_embedding'] = properties['submission_type'].apply(lambda x: embed_text(x, tokenizer, model))
    

    combined_features = np.hstack([
        properties['description_embedding'].tolist(),
        properties['location_embedding'].tolist(),
        properties['bedroom_embedding'].tolist(),
        properties['submission_type_embedding'].tolist()
    ])

    X = normalize(combined_features.astype('float32'))
    d = X.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(X)

    search_result = searchFAISSIndex(properties, query, index, nprobe=5, tokenizer=tokenizer, model=model, topk=topk)

    if not search_result.empty:
        search_result = search_result[['index', 'description', 'location', 'bedrooms', 'price.price', 'cosine_sim', 'L2_score']]
        return search_result.to_json(orient='records')
    else:
        return jsonify({"error": "No matching results found."})

if __name__ == '__main__':
    app.run(debug=True)

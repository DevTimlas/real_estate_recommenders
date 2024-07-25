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
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
import json

index = faiss.read_index('real_estate_final_index_new.faiss')
properties = pd.read_csv('modified_properties_new.csv')
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2', cache_dir='./cache_L')
model = AutoModel.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2', cache_dir='./cache_L')

def get_q(msg):
    llm = ChatOpenAI(temperature=0, model='gpt-4-turbo-2024-04-09',
                       api_key="sk-proj-pHkywA2mDUiHGcCWRM4TT3BlbkFJz3qlbeGgfG8C70iHX7WK")
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
    likewise for location too.
    extract the price and map it to price key, map the price key to unknown if it's not found in the sentence
        prices can be a number in full like 1000, 10,000, 1,000,000 or a short form such as 1K, 10K, 1M, 1B, etc 
        Take note of price modifiers such as less than, more than, for etc...
    extract the agent name and map it to agent key, map the agent key to unknown if it's not found in the sentence
    extract the property type and map it to property, map the property key to unknown if it's not found in the sentence
    """
    prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(sys_prompt),
                                               MessagesPlaceholder(variable_name="get_bed"),
                                               HumanMessagePromptTemplate.from_template(f"{msg}")])
    conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)

    memory.chat_memory.add_user_message(msg)
    response = conversation.invoke({"text": msg})
    return json.loads(response['text'])


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

def searchFAISSIndex(data, query, index, nprobe, tokenizer, model, topk=20):
    # Convert the query into embeddings for description, location, and bedroom
    description_embedding = embed_text(query['query'], tokenizer, model)
    location_embedding = embed_text(query['location'], tokenizer, model)
    bedroom_embedding = embed_text(query['bedroom'], tokenizer, model)
    sub_type_embedding = embed_text(query['sub_type'], tokenizer, model)
    prop_embedding = embed_text(query['property'], tokenizer, model)
    price_embedding = embed_text(query['price'], tokenizer, model)
    agent_embedding = embed_text(query['agent'], tokenizer, model)
    
    # Combine the embeddings
    query_embedding = np.hstack([description_embedding, location_embedding, bedroom_embedding, 
                                 sub_type_embedding, prop_embedding, price_embedding, agent_embedding])
    
    # Normalize the combined embedding
    dim = query_embedding.shape[0]
    query_embedding = query_embedding.reshape(1, dim)
    faiss.normalize_L2(query_embedding)
    
    # Set the number of probes
    index.nprobe = nprobe
    
    # Search the index
    D, I = index.search(query_embedding, topk)
    print("Distances:", D)
    print("Indices:", I)
    
    # Extract the indices and scores
    indices = I[0]
    L2_score = D[0]
    inner_product = [calculateInnerProduct(l2) for l2 in L2_score]
    # print("Inner products:", inner_product)
    # print("FAISS indices:", indices)
    
    # Retrieve rows using iloc
    matching_data = data.iloc[indices]
    # print("Matching data using iloc:", matching_data)
    
    # Create a DataFrame with the search results
    search_result = pd.DataFrame({
        'index': indices,
        'cosine_sim': inner_product,
        'L2_score': L2_score
    })
    # print("Search result DataFrame:", search_result)
    
    # Merge the search results with the original data
    dat = pd.concat([matching_data.reset_index(drop=True), search_result.reset_index(drop=True)], axis=1)    
    dat = dat.sort_values(by=['likes', 'views'], ascending=False)
    dat = dat.sort_values(by=['price.price'], ascending=True)
    dat[f"is_{query['location'].lower()}"] = dat['location'].str.lower() == query['location'].lower()
    dat[f"is_{query['sub_type'].lower()}"] = dat['submission_type'].str.lower() == query['sub_type'].lower()
    dat[f"is_{query['bedroom'].lower()}"] = dat['bedrooms'].str.contains(query['bedroom'], case=False, na=False)

    dat['bedrooms_numeric'] = dat['bedrooms'].str.extract('(\d+)', expand=False).astype(float)

    # Sort by is_lavington first, then by is_rent, then by location, and finally by submission type
    dat = dat.sort_values(by=[f"is_{query['location'].lower()}", 'is_rent', f"is_{query['bedroom'].lower()}", 'location', 'submission_type', 'bedrooms_numeric'], 
                          ascending=[False, False, False, True, True, True])

    
    return dat


@app.route('/search', methods=['POST'])
def search():
    inp = request.json
    # print(inp)
    query = get_q(inp.get("query"))
    print(query)
    topk = inp.get('k', 20)  # Get 'k' from the request, default to 20 if not provided

    search_result = searchFAISSIndex(properties, query, index, nprobe=5, tokenizer=tokenizer, model=model, topk=topk)
    search_result = search_result.drop(['description_embedding', 'location_embedding', 'bedroom_embedding', 'submission_type_embedding', 'property_type_embedding', 'price_embedding', 'agent_embedding'], axis=1)

    if not search_result.empty:
        # search_result = search_result[['index', 'description', 'location', 'bedrooms', 'price.price']]
        return search_result.to_json(orient='records')
    else:
        return jsonify({"error": "No matching results found."})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
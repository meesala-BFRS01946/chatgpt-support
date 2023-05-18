from flask import Flask, request, jsonify 
import requests
import re
from dotenv import load_dotenv
import openai
import json
import os
import logging
from datetime import datetime
from intents import intents
import translators as ts
import langchain
from langchain.llms import OpenAI, Cohere, HuggingFaceHub
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
import pandas as pd

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
gpt3 = OpenAI(model_name='text-davinci-003')
with open('phrases.json', 'r') as f:
    phrases = json.load(f)

app = Flask(__name__)
# set up logging
logs_folder = "logs"
if not os.path.exists(logs_folder):
    os.makedirs(logs_folder)
log_filename = logs_folder + "/app_" + datetime.now().strftime("%Y-%m-%d") + ".log"
logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


@app.route('/', methods=['POST'])
def pred():
    
    global text 
    data =request.get_json()
    text=data["text"]
    text=translate_to_english(text)
    mapp = json.dumps(phrases)
    prompt = " by taking the reference from the following map {} , please map the following text {} to the one of the intent in the following intents {} and return NULL if no intents are matched".format(mapp, text, intents)
    response = gpt3(prompt)
    intent = response.strip().lower()
    print(intent)
    logging.info("Received text: {}, Predicted intent: {}".format(text, intent))
    if is_number(text):
        return jsonify({"response":tracking_order(text),"intent":"awb_number"})
    elif intent == "tracking details":
        return jsonify({"response": "please enter your AWB number","intent":intent})
    elif intent == "greeting":
        return jsonify({"response": translate_to_english(greeting()),"intent":intent})
    elif intent == "greeting->greeting":
        return jsonify({"response": translate_to_english(greetingfallback()),"intent":intent})
    elif intent == "pickup not attempted":
        return jsonify({"response": "please enter your AWB number","intent":intent})
    
    res=train_doc(text)
    if res!="NULL":
        return jsonify({"response": train_doc(text),"intent":"LLM"})
    return jsonify({"response": create_ticket(text),"intent":"freshwork response"})

    

def tracking_order(awb):
    #awb="SRTP8501354758"
    tracking_url= os.getenv("TRACKING_URL").format(awb)
    response = requests.get(tracking_url)
    try:
        if response.status_code == 200:
            data = response.json()
            return str(data)
    except:
        return("Hi, we haven't found any AWB under this category.")
    

    
#@app.route('/greeting')
def greeting():
    prompt = "The user had got his query resolved in a e-commerce chatbot , please give reply to the following text as a e-commerce chatbot in english only {} ".format(text)
    response = gpt3(prompt)
    return str(response.strip())
def greetingfallback():
    prompt = "The user had got his query resolved in a e-commerce chatbot , please give reply to the following text as a e-commerce chatbot in english {} ".format(text)
    response = gpt3(prompt)
    return str(response.strip())


def translate_to_english(textt):
  try:
    result = ts.translate_text(textt, to_language='en')
    return result
  except:
    return textt
def is_number(input):
    pattern = r'\d{9,}'
    return bool(re.search(pattern, input))
    
def create_ticket(subject):
    freshwork_endpoint_url = os.getenv("FRESHWORK_ENDPOINT_URL")
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "subject": subject,
        "description": "text not handled by chatgpt",
        "status": 2,  # 2 is the status code for "Open"
        "priority": 1,  # 1 is the priority code for "Low"
        "email": "user@example.com"
    }
    key=os.getenv("FRESHWORKS_API_KEY")
    auth = (key, "x")
    response = requests.post(freshwork_endpoint_url, headers=headers, json=data, auth=auth)
    if response.status_code == 201:
        return "Ticket created successfully , Your Ticket Id : #{}".format(response.json()['id'])
    else:
        return "Failed to create ticket"

def train_doc(text):
    query=request.get_json()
   # loader = TextLoader("FAQ")
  #  documents = loader.load()
    loader = CSVLoader(file_path='./que_ans.csv',csv_args={'delimiter': ','})
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(batch_size=5), chain_type="map_reduce", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))
    user_q=text
    # query = "give the answer to the user's query .The answers should be given from the Document provided to you and if there is no answer from that document please return 'NULL'.Please ensure to provide the answer in bullet points.The user query is {}".format(user_q)
    query = "The user's query is:: {}. Generate a response based on the user's question. Remember that all the questions are going to be in context of the initial prompt only, respond with I'm not sure if any question is out of context but do not mislead the user.If the answer contains HTML code, convert it to human-readable format. Rephrase the response to make it more attractive, add spaces and line breaks wherever necessary, remove any \n or \t or any <> from your response, if the answer is in points try to list each point in a seperate line. And make sure to omit any references to image links. ENSURE that the meaning isnt lost!".format(user_q)
    return qa.run(query)



if __name__ == '__main__':
    app.run(debug=True)


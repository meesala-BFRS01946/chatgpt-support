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
from responses import responses
import langchain
from langchain.chains import APIChain
import translators as ts
from langchain.llms import OpenAI

load_dotenv()
openai.api_key =os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0)

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
    prompt = " by taking the reference from the following dictionary {} , please map the following text {} to the one of the intent in the following intents {} and return NULL if no intents are matched".format(mapp, text, intents)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.314,
        max_tokens=256,
        top_p=0.54,
        frequency_penalty=0.44,
        presence_penalty=0.17)
    intent = ((response.choices[0].text).strip()).lower()
    logging.info("Received text: {}, Predicted intent: {}".format(text, intent))
    if is_number(text):
        #print(tracking_order(text))
        return jsonify({"response":(tracking_order(text)),"intent":"awb_number","more_info":more_tracking_info(text)})
    elif intent == "tracking details":
        return jsonify({"response": tracking_no(),"intent":intent})
    elif intent == "greeting":
        return jsonify({"response": translate_to_english(greeting()),"intent":intent})
    elif intent == "greeting->greeting":
        return jsonify({"response": translate_to_english(greetingfallback()),"intent":intent})
    elif intent == "pickup not attempted":
        return jsonify({"response": "please enter your AWB number","intent":intent})
    # elif intent == "contact support":
    #     return jsonify({"response":"You can mail us on support@shiprocket.com","intent":intent})
    return jsonify({"response": create_ticket(text),"intent":"freshwork response"})

def ask_gpt(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.314,
        max_tokens=256,
        top_p=0.54,
        frequency_penalty=0.44,
        presence_penalty=0.17)
    return str((response.choices[0].text).strip()) 


def tracking_order(awb):
    #awb="SRTP8501354758"
    tracking_url= os.getenv("MORE_INFO")
    # headers={
    #     "Authorization":os.getenv("AUTH_TOKEN")
    # }
    chain_new = APIChain.from_llm_and_api_docs(llm,tracking_url, verbose=True)
    #response = requests.get(tracking_url,headers=headers)
    # try:
    #     if response.status_code == 200:
    #         data = response.json()
    #         r = data['tracking_data']['shipment_track'][0]['current_status']
    #         rr= responses["track_order"].format(r)
    #         return r
    # except:
    #     return("Hi, we haven't found any AWB under this category.")
    prompt= "what is the status of the following awb number {}".format(awb)
    r=chain_new.run(prompt)
    return r
    
def more_tracking_info(awb):
    url=os.getenv("MORE_INFO").format(awb)
    response=requests.get(url)
    return(str(response.json()))                                              
def awb_info(status):
    prompt = "share the current status as {} with the user and ask for any further queries".format(status)
    return ask_gpt(prompt)
def greeting():
    prompt = "Act as a support agent and pick a indian name and that will be your name when you speak to the user , please give greeting to the following  text in a empathitic way , as a e-commerce chatbot in english only {} ".format(text)
    return ask_gpt(prompt)
def greetingfallback():
    prompt = "The user had got his query resolved in a e-commerce chatbot , please give reply to the following text as a e-commerce chatbot in english {} ".format(text)
    return ask_gpt(prompt)
def tracking_no():
    prompt = "Ask the user for order tracking number in a friendly way"
    return ask_gpt(prompt)
def translate_to_english(textt):
  try:
    result = ts.translate_text(textt, to_language='en')
    return result
  except:
    return textt
def is_number(input):
    pattern = r'\d{8,}'
    return bool(re.search(pattern, input))
    
def create_ticket(subject):
    freshwork_endpoint_url = os.getenv("FRESHWORK_ENDPOINT_URL")
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "subject": subject,
        "description": "freshdesk",
        "status": 2,  # 2 is the status code for "Open" 
        "priority": 1,  # 1 is the priority code for "Low"
        "email": "user@example.com"
    }
    key=os.getenv("FRESHWORKS_API_KEY")
    auth = (key, "x")
    response = requests.post(freshwork_endpoint_url, headers=headers, json=data, auth=auth)
    if response.status_code == 201:
        return send_freshdesk_ticket(format(response.json()['id']))
    else:
        return "Failed to create ticket"
def track_ticket(ticket_id):
    status_dict = {
    2: 'Open',
    3: 'Pending',
    4: 'Resolved',
    5: 'Closed',
    6: 'Waiting on Customer',
    7: 'Waiting on Third Party'
                 }
    freshwork_endpoint_url = os.getenv("FRESHWORK_TRACK_ID_ENDPOINT_URL").format(ticket_id)
    headers = {
        "Content-Type": "application/json"
    }
    key=os.getenv("FRESHWORKS_API_KEY")
    auth = (key, "x")
    response = requests.get(freshwork_endpoint_url, headers=headers,auth=auth)
    if response.status_code == 201:
        return "The status of your ticket is => {}".format(status_dict[response.json()['status']])
    else:
        return "Failed to fetch the status of ticket"
def send_freshdesk_ticket(id):
    prompt = "Act as a support chatbot of a e-commerce company and convey that you have created a freshdesk ticket for the issue and its tracking id is following {}. make sure that your message is of only 2 lines".format(id)
    return ask_gpt(prompt)



if __name__ == '__main__':
    app.run(debug=True)


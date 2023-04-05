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
import translators as ts

load_dotenv()
openai.api_key = "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"

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
        return jsonify({"response":tracking_order(text),"intent":"awb_number"})
    elif intent == "tracking details":
        return jsonify({"response": "please enter your AWB number","intent":intent})
    elif intent == "greeting":
        return jsonify({"response": translate_to_english(greeting()),"intent":intent})
    elif intent == "greeting->greeting":
        return jsonify({"response": translate_to_english(greetingfallback()),"intent":intent})
    elif intent == "pickup not attempted":
        return jsonify({"response": "please enter your AWB number","intent":intent})
    elif intent == "contact support":
        return jsonify({"response":"You can mail us on support@shiprocket.com","intent":intent})
    return jsonify({"response": create_ticket(text),"intent":"freshwork response"})

    

def tracking_order(awb):
    #awb="SRTP8501354758"
    tracking_url= "https://apiv2.shiprocket.in/v1/tracking/{}".format(awb)
    response = requests.get(tracking_url)
    try:
        if response.status_code == 200:
            data = response.json()
            #r = data["tracking_data"]["shipment_track"][0]["current_status"]
            #rr= responses["track_order"].format(r)
            return str(data)
    except:
        return("Hi, we haven't found any AWB under this category.")
    

    
#@app.route('/greeting')
def greeting():
    prompt = "The user had got his query resolved in a e-commerce chatbot , please give reply to the following text as a e-commerce chatbot in english only {} ".format(text)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.314,
        max_tokens=256,
        top_p=0.54,
        frequency_penalty=0.44,
        presence_penalty=0.17)
    return str((response.choices[0].text).strip())
def greetingfallback():
    prompt = "The user had got his query resolved in a e-commerce chatbot , please give reply to the following text as a e-commerce chatbot in english {} ".format(text)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.314,
        max_tokens=256,
        top_p=0.54,
        frequency_penalty=0.44,
        presence_penalty=0.17)
    return str((response.choices[0].text).strip())


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
    url = "https://shiprocketdemo.freshdesk.com/api/v2/tickets"
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
    auth = ("Ifo1K69EBXPELX1DF5Sg", "x")
    response = requests.post(url, headers=headers, json=data, auth=auth)
    if response.status_code == 201:
        return "Ticket created successfully"
    else:
        return "Failed to create ticket"





if __name__ == '__main__':
    app.run(debug=True)


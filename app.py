from flask import Flask, request, jsonify , redirect , url_for , render_template
import requests
from dotenv import load_dotenv
import openai
import json
import os
import logging
from datetime import datetime
from intents import intents
from responses import responses

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
    #try:
    global text 
    data =request.get_json()
    text=data["text"]
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
    if intent == "tracking details":
        return jsonify({"response": tracking_order(),"intent":intent})
    elif intent == "greeting":
        return jsonify({"response": greeting(),"intent":intent})
        #return redirect("/greeting",var=text)
    elif intent == "greeting->greeting":
        return jsonify({"response": greetingfallback(),"intent":intent})
    elif intent == "pickup not attempted":
        return jsonify({"response": pickup_not_attempted(),"intent":intent})
    
    return jsonify({"response": intent})

    # except Exception as e:
    #     logging.exception("Error occurred: {}".format(str(e)))
    #     return jsonify({"intent": "error"})


#@app.route('/tracking')
def tracking_order():
    awb="SRTP8501354758"
    #awb=input("please enter the awb number")
    tracking_url= "https://apiv2.shiprocket.in/v1/tracking/{}".format(awb)
    token=os.getenv("bearer_token")
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(tracking_url)
    try:
        if response.status_code == 200:
            data = response.json()
            r = data["tracking_data"]["shipment_track"][0]["current_status"]
            rr= responses["track_order"].format(r)
            return rr
    except:
        return("Hi, we haven't found any AWB under this category.")
    
def pickup_not_attempted():
    awb_no="SRTP8501354758"
    tracking_url= "https://apiv2.shiprocket.in/v1/tracking/{}".format(awb_no)
    #token=os.getenv("bearer_token")
    #headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(tracking_url)
    # response_dict = response.json()
    # awbs = response_dict["data"]["awbs"]
    # awb_numbers = [awb["awb"] for awb in awbs]
    # if awb_no in awb_numbers:
    try:
        if response.status_code == 200:
            data = response.json()
            r = data["tracking_data"]["shipment_track"][0]["current_status"]
            rr= responses["track_order"].format(r)
            return rr
    except:
        return("Hi, we haven't found any AWB under this category.")

    
#@app.route('/greeting')
def greeting():
    prompt = "The user had got his query resolved in a e-commerce chatbot , please give reply to the following text as a e-commerce chatbot in english {} ".format(text)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.314,
        max_tokens=256,
        top_p=0.54,
        frequency_penalty=0.44,
        presence_penalty=0.17)
    return (response.choices[0].text).strip()
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
    return (response.choices[0].text).strip()





if __name__ == '__main__':
    app.run(debug=True)

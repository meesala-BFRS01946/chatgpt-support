import translators as ts
import re
import os
import requests

def translate_to_english(textt):
  try:
    result = ts.translate_text(textt, to_language='en')
    return result
  except:
    return textt

def is_number(input):
    pattern = r'\d{9,}'
    return bool(re.search(pattern, input))

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
    
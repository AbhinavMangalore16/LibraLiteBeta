import os
import requests
import json

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={os.getenv('GOOGLE_API_KEY')}"
headers = {"Content-Type": "application/json"}
data = {
    "contents": [
        {
            "parts": [
                {"text": "Explain how AI works in a few words"}
            ]
        }
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())

import ollama

import json
import csv
from langchain_ollama import OllamaLLM

# Open and read the JSON file
with open('ml_sentiments_absa.json', 'r') as file:
    reviews = json.load(file)


prompt_template =lambda  reviewBody:  f"""
[PRODUCT REVIEW\] {reviewBody}\n\n

[INSTRUCTION\] Respond to the above product review as if you were the product seller responding on the product review webpage with 1-2 sentences. Do not offer any refunds.
"""

model = OllamaLLM(model="llama3.1:8b")

file = open('review_responses.csv', 'w', newline ='')

with file:
    header = ['review_title', 'review_body', 'llm_response']
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    for review in reviews.values():
        reviewBody = review['reviewBody']
        reviewTitle = review['reviewTitle']
        prompt = prompt_template(reviewBody=reviewBody)
        print("prompt", prompt)
        response = model.invoke(prompt)
        print("================================LLM RESPONSE: ", response)
        
        writer.writerow({'review_title' : reviewTitle, 
                     'review_body':  reviewBody, 
                     'llm_response': response})
        

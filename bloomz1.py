import requests
import json
import random
import math

import logging

# Set up logging
logging.basicConfig(filename='output_with_passage.log', level=logging.INFO)

# initialise huggingface model and API key
API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
# headers = {"Authorization": "Bearer API_KEY"}



def query(payload, golden_answer):
    response = requests.post(API_URL, headers=headers, json=payload)
    output = response.json()
    if isinstance(output, dict) and "error" in output.keys():
        # handle error
        print(output)
        payload["inputs"] = reduce_prompt_size(payload["inputs"])
        score = query(payload, golden_answer)
        return score
    else:
        if output[0]["generated_text"] == "":
            print(output)
            return 0
            
        result = output[0]["generated_text"].split()[0]
        score = 1 if result == str(golden_answer) else 0
        print(f'Score: {score}\n')
        print(payload["inputs"])
        print(f'Model: {result}, GOlden Answer: {golden_answer}')
        return score

# Bloomz models take a max token size of 1000
# the below function truncates if the size of the prompt exceeds 1000 tokens
  
def reduce_prompt_size(prompt):
    updated_prompt = "Question: ".join(prompt.split("Question: ")[:-2])
    updated_prompt += "Question: "+prompt.split("Question: ")[-1]
    return updated_prompt


def import_data(file_path):
    data = []
    # Open the file in read mode
    with open(file_path, 'r') as f:
        # Load the JSON data from the file
        for line in f:
            data.append(json.loads(line))
    return data

data = import_data('dev.jsonl')
selected_propmts = []

score = 0
iterations = 100

for i in range(iterations):
    # prompt generation
    prompt = ""
    for a in range(2):
        selected = True
        while selected:
            random_val = math.floor(random.random() * len(data))
            if random_val in selected_propmts or data[random_val]["answer"] == True:
                selected_propmts.append(random_val)
                prompt += f'Question: {data[random_val]["question"]} \nPassage: {data[random_val]["passage"]} \nAnswer: {data[random_val]["answer"]}\n'
                selected = False
    
        selected = True
        while selected:
            random_val = math.floor(random.random() * len(data))
            if random_val in selected_propmts or data[random_val]["answer"] == False:
                selected_propmts.append(random_val)
                prompt += f'Question: {data[random_val]["question"]} \nPassage: {data[random_val]["passage"]} \nAnswer: {data[random_val]["answer"]}\n'
                selected = False
            
    test_data = {}
    selected = True
    
    while selected:    
        random_val = math.floor(random.random() * len(data))
        if random_val in selected_propmts:
            continue
        test_data = { "question": data[random_val]["question"],  "answer": data[random_val]["answer"], "passage": data[random_val]["passage"] }

        prompt += f'Question: {test_data["question"]} \nPassage: {data[random_val]["passage"]} \nAnswer: '
        # print(reduce_prompt_size(prompt))
        score += query({"inputs": prompt,  "parameters": {"return_full_text": False}}, test_data["answer"])
        print(f'Running Accuracy after {i+1} iterations: {(score/(i+1))*100}')
        selected = False
    
print(f'Final Accuracy: {score/iterations * 100}')
logging.info(f'Final Accuracy: {score/iterations * 100}')

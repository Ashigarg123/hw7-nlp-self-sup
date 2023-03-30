import os
import openai
import random
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM


## create prompt using balanced demonstrations

def get_demos():

    print("Loading the dataset ...")
    dataset = load_dataset("boolq")
    print("Dataset loaded successfully!")
    dataset = dataset.shuffle()  # shuffle the data

    ## get equal demonstrations 
    #print(dataset['train'][0])
    rng = np.random.default_rng()
    indices = rng.integers(low=0, high=len(dataset['train']), size=50)
    no_true_samples = 0 
    no_false_samples = 0 
    true_samples_store = []
    false_samples_store = []

    for i in indices:
        curr = dataset['train'][int(i)]
        if curr['answer'] is False and no_false_samples < 4:
            false_samples_store.append(curr)
            no_false_samples +=1 
        elif curr['answer'] and no_true_samples < 4:
            true_samples_store.append(curr)
            no_true_samples +=1 

    all_samples = true_samples_store +  false_samples_store
    random.shuffle(all_samples)
    ## test samples 
    test_indices = rng.integers(low=0, high=len(dataset['validation']), size=100) 
    all_test_samples = [dataset['validation'][int(i)] for i in test_indices]

    return all_samples, all_test_samples



def res_api(prompt_):
    openai.api_key = "API KEY HERE"
    response = openai.Completion.create(
               model="davinci",
            prompt= prompt_,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
    temp_ans = response['choices'][0]['text'] 
    if '\n' in temp_ans:
        temp_ans = temp_ans[:temp_ans.index('\n')]

    return temp_ans

def bloomz_api(prompt_):

    my_tokenizer=AutoTokenizer.from_pretrained('bigscience/bloomz')
    model = AutoModelForCausalLM.from_pretrained('bigscience/bloomz')
    #inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt")
    inputs = my_tokenizer.encode(prompt_, return_tensors="pt")
    outputs = model.generate(inputs) 
    temp_ans = my_tokenizer.decode(outputs[0])

    if '\n' in temp_ans:
        temp_ans = temp_ans[temp_ans.rindex('\n')+1:]
    
    vals =  ['true', 'yes']
    final_ans = any(x in temp_ans.lower() for x in vals )

    return final_ans

def eval(all_samples, all_test_samples):

    prompt = ""
    for sample_ in all_samples:
        prompt = "question:" + sample_['question'] + "\n" + "passage:" + sample_['passage'] + "\n" + "answer:" + "true" if sample_['answer'] else "False" + "\n" 
    total = len(all_test_samples)
    right_pred = 0
    for sample in all_test_samples:
        test_prompt = prompt + "question:" + sample['question'] + "\n" + "passage:" + sample['passage'] + "\n" + "answer:"
        ##get response 
        openai_res = bloomz_api(test_prompt)
        ## compare preds 
        #res: falsequestion:is mac from shark tale in air between us
        # res = 'true' in openai_res.lower()
        # print("res: {}".format(res))
        if openai_res == sample['answer']:
            right_pred +=1 
    
    acc = right_pred/total * 100 
    return acc

# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    all_samples, all_test_samples = get_demos()
    # print(all_test_samples)
    accuracy = eval(all_samples, all_test_samples)
    print("Accuracy:{}".format(accuracy))




    



import json
import sys
import argparse
import os
from openai_query import chat, davinci_run
from bing_search import query_bing_return
import pandas as pd
import logging
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
import re


def process_search_query(s):
    s = s.strip().strip("\"").strip("“").strip("”").strip("’").strip("‘").strip("\"").strip(".").strip()
    return s

def extract_answers(ans,is_pair=False):
    ans_lis = ans.split("\n")
    gen_pubs = []
    for line in ans_lis:
        if re.search(r"\d+.",line):
            line_new = re.sub(r'\d+.',"",line,count=1)
            gen_pubs.append(line_new)
    return gen_pubs

def reference_query(title):
    global model_name
    global LOG_PATH

    temperature = 1.0
    n = 1
    num_titles = 5
    log_results = {}
    log_results["title"] = title
    alias = title.replace(" ","_").replace(":","").replace("/","_")

    logging.basicConfig(filename=LOG_PATH + alias + ".log",filemode='w',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',force=True)
    logging.info("model_name: " + model_name + " temperature: " + str(temperature))  

    if model_name == "gpt-4" or model_name == "gpt-3.5-turbo" or model_name == "gpt-35-turbo":

        messages=[{"role": "system", "content" : "You are a helpful assistant"}]
        query = {"role": "user", "content" : 'List {} existing references related to "{}".  Just output the titles. Do not mention the authors. Output format should be <num.> <title>'.format(num_titles,title)}
        messages.append(query)

        logging.info("Sending query\n {}".format(messages))    
        [ans] = chat(model_name,messages,temperature,n=n)
        logging.info("Model Answer\n {}".format(ans)) 
        log_results["model_answer_main_query"] = ans

    else:
        prompt = 'List {} existing references related to "{}".  Just output the titles. Do NOT mention the authors. Output format should be <num.> <title>'.format(num_titles,title)
        logging.info("Sending query\n {}".format(prompt))
        ans = davinci_run(model_name,prompt,temperature,n=n)[0]
        logging.info("Model Answer\n {}".format(ans))
        log_results["model_answer_main_query"] = ans
    
    gen_list = extract_answers(ans)
    log_results["gen_list"] = gen_list
  
    logging.info("Extracted Answers\n {}".format(gen_list))  
    return log_results


def main():
    global model_name
    global read_path
    global WRITE_JSON_PATH
    global start
    global num
    
    title_list = open(read_path,"r").readlines()
    res = []
    st_index = start
    if st_index > 0:
        try:
            res = json.load(open(WRITE_JSON_PATH,"r"))
        except:
            print("No file found, starting from scratch from the starting index of the concept provided")
    counter = 0
    for i,entry in enumerate(title_list[st_index:]):
        if num != -1 and counter >= num:
            break
        counter += 1
        title = entry.strip()
        print("Processing title: {} index: {}".format(title,i+st_index))
        log_results = reference_query(title)
        res.append(log_results)
            #Writing at every 20th iteration in case API crashes
        if (i+1)%20 == 0:
            json.dump(res,open(WRITE_JSON_PATH,"w"),indent=2,ensure_ascii=False)

    json.dump(res,open(WRITE_JSON_PATH,"w"),indent=2,ensure_ascii=False)

def convert_to_csv():
    global WRITE_JSON_PATH
    global WRITE_DF_PATH
    concept_lis = json.load(open(WRITE_JSON_PATH,"r"))
    res = []
    for i,concept in enumerate(concept_lis):
        for j,gen_title in enumerate(concept["gen_list"]):
            dict = {}
            gen_title = process_search_query(gen_title)
            dict["gen_title"] = gen_title
            dict["title"] = concept["title"]
            dict["model_answer_main_query"] = concept["model_answer_main_query"]
            #bing query
            dict["bing_return"],_ = query_bing_return(gen_title)
            print(i,j,"done")
            res.append(dict)
    df = pd.DataFrame(res)
    df.to_csv(WRITE_DF_PATH,index=False)
    with open(WRITE_JSON_PATH,"w") as f:
        json.dump(concept_lis,f,indent=2,ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="gpt-4", help="model name")
    parser.add_argument("--start", "-s", type=int, default=0, help="which to start at")
    parser.add_argument("--num", "-n", type=int, default=-1, help="How many to do, -1 means all")  
    parser.add_argument("--read_path", "-r", type=str, default="acm_ccs_200.titles", help="read path")
    args = parser.parse_args()
    model_name = args.model_name
    read_path = args.read_path
    start = args.start
    num = args.num



    log_dir = "logs_1000_acm_"+model_name
    LOG_PATH = "logs/{}/".format(log_dir)
    os.makedirs(LOG_PATH,exist_ok=True)
    WRITE_JSON_PATH = LOG_PATH + model_name+"_acm.json"
    WRITE_DF_PATH = LOG_PATH + model_name+"_acm.csv"
    main()
    convert_to_csv()
import json
import pandas as pd
import numpy as np
import sys
from openai_query import chat, davinci_run
import time
import logging
import os
import argparse
import shutil



def process_search_query(s):
    s = s.strip().strip("\"").strip("“").strip("”").strip("’").strip("‘").strip("\"").strip(".").strip()
    return s

def neural_agreement(stmt,prompt_type,i=None,full_ans=None,res1=None,res2=None,res3=None):
    global model_name
    global LOG_PATH


    stmt = process_search_query(stmt)

    temperature = 1.0
    n = 10

    alias = (stmt+"_{}prompt_type").format(prompt_type).replace(" ","_").replace("\\","").replace("/","").replace("?","").replace("!","").replace(",","").replace(":","").replace(";","")
    logging.basicConfig(filename=LOG_PATH + alias + ".log",filemode='w',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',force=True)
   
    logging.info("model_name: " + model_name + " temperature: " + str(temperature))
    logging.info("query stmt: " + stmt)
    logging.info("prompt_type: " + str(prompt_type))
    if type(res1) != str:
        res1 = "<CONTENT FILTERED>"
    if type(res2) != str:
        res2 = "<CONTENT FILTERED>"
    if type(res3) != str:
        res3 = "<CONTENT FILTERED>"
    
    if model_name == "gpt-4" or model_name == "gpt-3.5-turbo" or model_name == "gpt-35-turbo":   
        if prompt_type == 2:
            prompt = "Does the reference \"{}\" exist? Just output yes/no.".format(stmt)
            messages=[{"role": "system", "content" : "You are a helpful assistant"}]
            query = {"role": "user", "content" : prompt}
            messages.append(query)

        if prompt_type == 3:
            messages=[{"role": "system", "content" : "You are a helpful assistant"}]
            query = [{"role": "user", "content" : "Give a famous reference for reading."},{"role": "assistant", "content" : '"{}"'.format(stmt)},{"role": "user", "content" : "Does the above paper exist? Just output yes/no."}]
            messages.extend(query)

        if prompt_type == 4:
            messages=[{"role": "system", "content" : "You are a helpful assistant"}]
            query = [{"role": "user", "content" : "A language model generated references related to a research topic with the following titles:\n{}\nDoes the reference with title #{} exist? Just output yes/no.".format(full_ans,(i%5)+1)}]

            messages.extend(query)

        logging.info("Sending query\n {}".format(messages))       
        n_completion = chat(model_name,messages,temperature=temperature,n=n, max_tokens=15)

    else:
        if prompt_type == 2:
            prompt = "Does the reference \"{}\" exist? Just output yes/no.".format(stmt)

        if prompt_type == 3:
            prompt = 'Give a famous reference for reading.\n"{}"\nDoes the above paper exist? Just output yes/no.'.format(stmt)

        if prompt_type == 4:
            prompt = "A language model generated references related to a research topic with the following titles:\n{}\nDoes the reference with title #{} exist? Just output yes/no.".format(full_ans,(i%5)+1)

        logging.info("Sending query\n {}".format(prompt))

        n_completion = davinci_run(model_name,prompt,temperature=temperature,n=n,max_tokens=15)

    n_prob = 0
    ans_lis = []
    for i in range(n):
            n_ans =  n_completion[i]
            logging.info("Received Model response{}\n {}".format(i,n_ans))
            ans_lis.append(n_ans)
            if "yes" in n_ans.lower():
                n_prob = n_prob + 1
    n_prob = n_prob/n
    return ans_lis,n_prob


def iterate_for_neural_agreement(prompt_type, st_index, num, WRITE_DF_PATH):

    counter = 0
    
    #for i,row in df_valid.iterrows():
    df = pd.read_csv(WRITE_DF_PATH)

    for i,row in df.iterrows():
        if i < st_index:
            continue
        if num != -1 and counter >= num:
            break
        counter += 1
        
        if prompt_type == 2:
            ans_list,n_prob = neural_agreement(row["gen_title"],2)
            df.loc[i,"neural_ans2_list"] = str(ans_list)
            df.loc[i,"neural_ans2_prob"] = n_prob
        elif prompt_type == 3:
            ans_list,n_prob = neural_agreement(row["gen_title"],3)
            df.loc[i,"neural_ans3_list"] = str(ans_list)
            df.loc[i,"neural_ans3_prob"] = n_prob
 
        elif prompt_type == 4:
            ans_list,n_prob = neural_agreement(row["gen_title"],4,i,row["model_answer_main_query"])
            df.loc[i,"neural_ans4_list"] = str(ans_list)
            df.loc[i,"neural_ans4_prob"] = n_prob
        
        print(i,"done in method", prompt_type)
        if i % 20 == 0:
            df.to_csv(WRITE_DF_PATH,index=False)
            print(i,"saved")

    df.to_csv(WRITE_DF_PATH,index=False)

if __name__ == "__main__":
    # read arguments

    parser = argparse.ArgumentParser()
    # add required filename positional argument
    parser.add_argument("--start", "-s", type=int, default=0, help="which to start at")
    parser.add_argument("--num", "-n", type=int, default=-1, help="How many to do, -1 means all")  
    parser.add_argument("--out_file", "-o", type=str, default="")
    parser.add_argument("--model_name", "-m", type=str, default="gpt-4")
    parser.add_argument("--methods", "-md", type=str, default="234", help="Which methods to do, eg 23 or 2 or 234")
    args = parser.parse_args()

    model_name = args.model_name
    log_dir = "logs_1000_acm_"+model_name
    LOG_PATH = "logs/{}/neural_agreement_logs_{}/".format(log_dir,model_name)
    WRITE_DF_PATH = "logs/{}/{}_acm.csv".format(log_dir,model_name) 
    os.makedirs(LOG_PATH,exist_ok=True)

    if args.out_file != "":
        assert not os.path.exists(args.out_file), "Output file already exists"
        # copy file from WRITE_DF_PATH to args.out_file
        shutil.copyfile(WRITE_DF_PATH, args.out_file)
        WRITE_DF_PATH = args.out_file

    for m in range(2, 5):
        if str(m) in args.methods:            
            print("**** doing method", m)
            iterate_for_neural_agreement(m, args.start, args.num, WRITE_DF_PATH)

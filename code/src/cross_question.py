import json
import pandas as pd
import numpy as np
import sys
from openai_query import chat, davinci_run
import time
import re
import logging
import os
import argparse
import shutil


def cross_question(title,i):
    global model_name
    global LOG_PATH

    temperature = 1.0
    n = 3
    alias = (title[:15]+str(i)).replace(" ","_").replace("/","")

    logging.basicConfig(filename=LOG_PATH + alias + ".log",filemode='w',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',force=True)

    logging.info("model_name: " + model_name + " temperature: " + str(temperature))  

    who_list = []
    if model_name == "gpt-4" or model_name == "gpt-3.5-turbo" or model_name == "gpt-35-turbo":
        independent_query =  [{"role": "system", "content" : "You are a helpful assistant"}]
        query = {"role": "user", "content" : 'Who were the authors of the reference, "{}"? Please, list only the author names, formatted as - AUTHORS: <firstname> <lastname>, separated by commas. Do not mention the reference in the answer.\nAUTHORS:'.format(title)}
        independent_query.append(query)

        logging.info("Sending query\n {}".format(independent_query))    
        completion = chat(model_name,independent_query,temperature,n=n)
        for i in range(n):
            ans = completion[i]
            logging.info("Model Answer {}:{}".format(i,ans))
            who_list.append(ans)

    else:
        prompt = 'Who were the authors of the reference, "{}"? Please, list only the author names, formatted as - AUTHORS: <firstname> <lastname>, separated by commas. Do not mention the reference in the answer.\nAUTHORS:'.format(title)
        logging.info("Sending prompt\n {}".format(prompt))
        who_list = davinci_run(model_name,prompt,temperature,n=n)
        logging.info("Model Answer:{}".format(who_list))
        
    return who_list

def main():
    global WRITE_DF_PATH
    global start
    global num

    st_index = start
    counter = 0

    df = pd.read_csv(WRITE_DF_PATH)
    #TODO, add json reading
    for i,row in df.iterrows():
        if i < st_index:
            continue
        if args.num != -1 and counter >= args.num:
            break
        counter += 1
        who_list = cross_question(row["gen_title"],i)
        print(i,"done")
        #add column to df about agreement
        if len(who_list) < 3:
            df.loc[i,"valid_entry"] = False
        else:
            df.loc[i,"model_ans_1"] = who_list[0]
            df.loc[i,"model_ans_2"] = who_list[1]
            df.loc[i,"model_ans_3"] = who_list[2]
        if i % 20 == 0:
            df.to_csv(WRITE_DF_PATH,index=False)
            print(i,"saved")
    df.to_csv(WRITE_DF_PATH,index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", "-s", type=int, default=0, help="which to start at")
    parser.add_argument("--num", "-n", type=int, default=-1, help="How many to do, -1 means all")  
    parser.add_argument("--model_name", "-m", type=str, default="gpt-4")
    parser.add_argument("--out_file", "-o", type=str, default="")
    args = parser.parse_args()

    model_name = args.model_name
    start = args.start
    num = args.num
    log_dir = "logs_1000_acm_"+model_name
    LOG_PATH = "logs/{}/cross_question_logs_{}/".format(log_dir,model_name)
    WRITE_DF_PATH = "logs/{}/{}_acm.csv".format(log_dir,model_name) 
    os.makedirs(LOG_PATH,exist_ok=True)
    
    if args.out_file != "":
        assert not os.path.exists(args.out_file), "Output file already exists"
        # copy file from WRITE_DF_PATH to args.out_file
        shutil.copyfile(WRITE_DF_PATH, args.out_file)
        WRITE_DF_PATH = args.out_file
    main()
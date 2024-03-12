import sys
sys.path.append('<llama repo cloned path>')
from typing import Optional
import fire
import llama.generation as gen
import json
import argparse
import os
from bing_search import query_bing_return_mkt
import pandas as pd
import logging
import numpy as np
import re
import multiprocessing
from prompts import *
import time
import ast


def process_search_query(s):
    s = s.strip().strip("\"").strip("“").strip("”").strip("’").strip("‘").strip("\"").strip(".").strip()
    return s

def extract_answers(ans,is_pair=False):
    ans_lis = ans.split("\n")
    gen_pubs = []
    for line in ans_lis:
        if re.search(r"\d+\.",line):
            #print(line)
            match_pub = re.search(r'(\d+)\.\s*(?:“|")(.+)(?:"|”)',line)
            line_new = match_pub.group(2) #re.sub(r'\d+.',"",line,count=1)
            gen_pubs.append(line_new)
    return gen_pubs


def extract_authors_from_ans(title,ans):
    ans_lis = ans.split("\n")
    authors = None
    authors = ans.replace(title,"provided reference").replace("AUTHORS:","")
   
    return authors

def df_extract_authors(path):
    df = pd.read_csv(path)
    df["IQ_ans1"] = df.apply(lambda x : extract_authors_from_ans(x["gen_title"],x["IQ_full_ans1"]),axis=1)
    df["IQ_ans2"] = df.apply(lambda x : extract_authors_from_ans(x["gen_title"],x["IQ_full_ans2"]),axis=1)
    df["IQ_ans3"] = df.apply(lambda x : extract_authors_from_ans(x["gen_title"],x["IQ_full_ans3"]),axis=1)
    df.to_csv(path,index=False)

def reference_query(generator,prompt,concept,max_gen_len,top_p,temperature,LOG_PATH):
    log_results = {}
    log_results["title"] = concept
    alias = concept.replace(" ","_").replace(":","").replace("/","_")

    logging.basicConfig(filename=LOG_PATH + alias + ".log",filemode='w',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',force=True)
    logging.info("temperature: " + str(temperature)) 

    dialogs = [prompt(concept)]

    logging.info("Sending query\n {}".format(dialogs))    
    
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        top_p = top_p,
        temperature = temperature
    )
    
    #there is only one dialog, we are not batching right now
    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            logging.info(f"{msg['role'].capitalize()}: {msg['content']}\n")
        logging.info(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
        logging.info("\n==================================\n")
    
        #logging.info("Model Answer\n {}".format(ans)) 
        log_results["model_answer_main_query"] = result['generation']['content']
        gen_list = extract_answers(result['generation']['content'])
        log_results["gen_list"] = gen_list
        logging.info("Extracted Answers\n {}".format(gen_list))  
    
    return log_results

def direct_query(generator,prompt,title,max_gen_len,top_p,temperature,LOG_PATH,i=None,all_ans=None):
    
    if i is not None and all_ans is not None:
        assert title in all_ans
        prompt = prompt(all_ans,(i%5)+1)
    else:
        prompt = prompt(title)
    
    alias = title.replace(" ","_").replace(":","").replace("/","_")

    logging.basicConfig(filename=LOG_PATH + alias + ".log",filemode='w',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',force=True)
    logging.info("temperature: " + str(temperature)) 

    dialogs = [prompt]

    logging.info("Sending query\n {}".format(dialogs))    
    
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        top_p = top_p,
        temperature = temperature
    )
    
    #there is only one dialog, we are not batching right now
    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            logging.info(f"{msg['role'].capitalize()}: {msg['content']}\n")
        logging.info(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
        logging.info("\n==================================\n")
        logging.info("Yes probability: {}".format(result['yes_prob']))
        logging.info("Token with respect to which prob is calculated (0) mean default\n {}".format(result['prob_token']))
        #logging.info("Model Answer\n {}".format(ans))
    print(result['generation']['content'],result['yes_prob'].item())
    return result['generation']['content'],result['yes_prob'].item(),result['prob_token']

def DQ_query_sample(generator,prompt,gen_title,max_gen_len,top_p,temperature,LOG_PATH,i=None,all_ans=None,num_gen=None):
    if i is not None and all_ans is not None:
        assert gen_title in all_ans
        prompt = prompt(all_ans,(i%5)+1)
    else:
        prompt = prompt(gen_title)
    alias = gen_title.replace(" ","_").replace(":","").replace("/","_")

    logging.basicConfig(filename=LOG_PATH + alias + ".log",filemode='w',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',force=True)
    logging.info("temperature: " + str(temperature)) 

    dialogs = [prompt]

    logging.info("Sending query\n {}".format(dialogs))    
    
    model_ans = []
    for j in range(num_gen):
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            top_p = top_p,
            temperature = temperature
        )
    
        #there is only one dialog, we are not batching right now
        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                logging.info(f"{msg['role'].capitalize()}: {msg['content']}\n")
            logging.info(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
            logging.info("\n==================================\n")
            #ans = extract_authors_from_ans(result['generation']['content'])
            ans = result['generation']['content']
            
            #logging.info("Extracted Model Answer\n {}".format(ans))
            model_ans.append(ans)
    
    
    n_prob = 0
    ans_lis = []
    for i in range(num_gen):
            n_ans =  model_ans[i]
            logging.info("Received Model response{}\n {}".format(i,n_ans))
            ans_lis.append(n_ans)
            if "yes" in n_ans.lower():
                n_prob = n_prob + 1
    n_prob = n_prob/num_gen
    #return ans_lis,n_prob
        
    logging.info("Model n_prob\n {}".format(n_prob))
    return model_ans, n_prob

def correct_sample_dq(file_path):
    df = pd.read_csv(file_path)
    def return_prob(x):
        n_prob = 0
        num_gen = len(x)
        for i in range(num_gen):
            if "yes" in x[i].lower():
                n_prob = n_prob + 1
        n_prob = n_prob/num_gen
        return n_prob
    df["DQ1_prob_sample"] = df["DQ1_ans_sample"].apply(lambda x: return_prob(ast.literal_eval(x)))
    df["DQ2_prob_sample"] = df["DQ2_ans_sample"].apply(lambda x: return_prob(ast.literal_eval(x)))
    df["DQ3_prob_sample"] = df["DQ3_ans_sample"].apply(lambda x: return_prob(ast.literal_eval(x)))
    df.to_csv(file_path,index=False)
    
def IQ_query(generator,prompt,num_gen,gen_title,max_gen_len,top_p,temperature,LOG_PATH):
    alias = gen_title.replace(" ","_").replace(":","").replace("/","_")

    logging.basicConfig(filename=LOG_PATH + alias + ".log",filemode='w',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',force=True)
    logging.info("temperature: " + str(temperature)) 

    dialogs = [prompt(gen_title)]

    logging.info("Sending query\n {}".format(dialogs))    
    
    model_ans = []
    for j in range(num_gen):
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            top_p = top_p,
            temperature = temperature
        )
    
        #there is only one dialog, we are not batching right now
        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                logging.info(f"{msg['role'].capitalize()}: {msg['content']}\n")
            logging.info(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
            logging.info("\n==================================\n")
            #ans = extract_authors_from_ans(result['generation']['content'])
            ans = result['generation']['content']
            #logging.info("Extracted Model Answer\n {}".format(ans))
            model_ans.append(ans)
        
    logging.info("Model Answer\n {}".format(model_ans))
    return model_ans
        
 

def main_DQ(
    ckpt_dir: str,
    tokenizer_path: str,
    num_gen: int = None,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    read_path: str = None,
    LOG_PATH: str = None,
    start_index: int = None,
    how_many: int = None, #-1 for all
    dq_type: int = None
):
    assert start_index is not None
    assert how_many is not None
    

    generator = gen.Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size)
    
    os.makedirs(LOG_PATH, exist_ok=True)  
    df = pd.read_csv(read_path)
    suffix = ""
    if num_gen is not None:
        suffix = "_sample"
    if dq_type == 1:
        prompt = prompt_DQ1 if num_gen is None else prompt_DQ1_sample
        verbose_ans = f"DQ1_ans{suffix}"
        prob_ans = f"DQ1_prob{suffix}"
        prob_ans_token = "DQ1_prob_token"
    elif dq_type == 2:
        prompt = prompt_DQ2 if num_gen is None else prompt_DQ2_sample
        verbose_ans = f"DQ2_ans{suffix}"
        prob_ans = f"DQ2_prob{suffix}"
        prob_ans_token = "DQ2_prob_token"
    elif dq_type == 3: #need to check some implementations
        prompt = prompt_DQ3 if num_gen is None else prompt_DQ3_sample
        verbose_ans = f"DQ3_ans{suffix}"
        prob_ans = f"DQ3_prob{suffix}"
        prob_ans_token = "DQ3_prob_token"
    counter = 0
    for i,row in df.iterrows():
        if i < start_index:
            continue
        if how_many != -1 and counter >= how_many:
            break
        counter += 1
        
        if num_gen is None:
            if dq_type == 1 or dq_type == 2:
                ans,prob,prob_token = direct_query(generator,prompt,row["gen_title"],max_gen_len,top_p,temperature,LOG_PATH)
            
            if dq_type == 3:
                ans,prob,prob_token = direct_query(generator,prompt,row["gen_title"],max_gen_len,top_p,temperature,LOG_PATH,i,row["model_answer_main_query"])
        
        else:
            if dq_type == 1 or dq_type == 2:
                ans,prob = DQ_query_sample(generator,prompt,row["gen_title"],max_gen_len,top_p,temperature,LOG_PATH,i=None,all_ans=None,num_gen=num_gen)
            if dq_type == 3:
                ans,prob = DQ_query_sample(generator,prompt,row["gen_title"],max_gen_len,top_p,temperature,LOG_PATH,i,row["model_answer_main_query"],num_gen=num_gen)
            
            
        df.loc[i,verbose_ans] = str(ans)
        df.loc[i,prob_ans] = prob
        if num_gen is None:
            df.loc[i,prob_ans_token] = prob_token
        
        print(i,"done in method", dq_type)
        if i % 20 == 0:
            df.to_csv(read_path,index=False)
            print(i,"saved")
    df.to_csv(read_path,index=False)

def main_IQ(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.0,
    num_gen: int = 1,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    read_path: str = None,
    LOG_PATH: str = None,
    start_index: int = None,
    how_many: int = None, #-1 for all
):
    assert start_index is not None
    assert how_many is not None
    
    generator = gen.Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size)
    
    os.makedirs(LOG_PATH, exist_ok=True)  
    df = pd.read_csv(read_path)
    
    counter = 0
    for i,row in df.iterrows():
        if i < start_index:
            continue
        if how_many != -1 and counter >= how_many:
            break
        counter += 1
        #(generator,prompt,num_gen,gen_title,max_gen_len,top_p,temperature,LOG_PATH)
        res = IQ_query(generator,prompt_IQ,num_gen,row["gen_title"],max_gen_len,top_p,temperature,LOG_PATH)       
        for j in range(num_gen):    
            df.loc[i,"IQ_full_ans{}".format(j+1)] = res[j]
        
        print(i,"done in method")
        if i % 20 == 0:
            df.to_csv(read_path,index=False)
            print(i,"saved")
    df.to_csv(read_path,index=False)
 
def get_agreement_frac(s: str):
    "Extract agreement percentage from string and return it as a float between 0 and 1"
    x = s.strip().upper()
    if x.startswith("ANSWER"):
        x = x[6:].strip()
    if x.startswith("ANS"):
        x = x[3:].strip()
    x = x.strip(": ")
    # match an initial integer or float
    m = re.match(r"^[0-9]+\.?[0-9]*", x)
    if not m:
        return 0.0
    return min(float(m.group(0)) / 100.0, 1.0) 
 
def consistency_check_pair(list1,list2,generator):
    
    PROMPT = """Below are what should be two lists of authors. On a scale of 0-100%, how much overlap is there in the author names (ignore minor variations such as middle initials or accents)? Answer with a number between 0 and 100. Also, provide a justification. Note: if either of them is not a list of authors, output 0. Output format should be ANS: <ans> JUSTIFICATION: <justification>.
    
    1. <NAME_LIST1>
    2. <NAME_LIST2>"""
    
    list1 = str(list1).strip().replace("\n", " ")
    list2 = str(list2).strip().replace("\n", " ")
    prompt = PROMPT.replace("<NAME_LIST1>", list1).replace("<NAME_LIST2>", list2)
    messages = [[{"role": "user", "content": prompt}]]
    #print("messages", messages)
    
    results = generator.chat_completion(
        messages,  # type: ignore
        max_gen_len=150,
        top_p = 0.95,
        temperature = 0.0
    )
    ans = results[0]["generation"]["content"]
    return (get_agreement_frac(ans), ans)    

def consistency_check(auth_lists,generator):
    n = len(auth_lists)
    assert n >= 2
    records = []
    fracs = []
    for i in range(n):
        for j in range(i):
            frac,ans = consistency_check_pair(auth_lists[i],auth_lists[j],generator)
            records.append(ans)
            fracs.append(frac)
            # print("auth_lis1",auth_lists[i])
            # print("auth_lis2",auth_lists[j])
            # print("frac",frac)
            # print("ans",ans)
            # print("================")
    mean = sum(fracs)/len(fracs)
    for a in auth_lists:
        print(" ",a)
    print("mean",mean)
    print()
    return mean, records
    

def main_CC(ckpt_dir: str,
    tokenizer_path: str,
    read_path: str = None,
    LOG_PATH: str = None,
    start_index: int = None,
    how_many: int = None, #-1 for all
    ):
    generator = gen.Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=512,
            max_batch_size=1)
    
    start = time.time()
    log_f = open(LOG_PATH, "a")
    df = pd.read_csv(read_path)
    counter = 0
    for i, row in df.iterrows():
        if i < start_index:
            continue
        if how_many != -1 and counter >= how_many:
            break
        counter += 1
        
        mean, records = consistency_check([df.loc[i, f"IQ_ans{j}"] for j in "123"],generator)      

        log_f.write("title:" + row["gen_title"]+"\n")
        log_f.write("mean:" + str(mean)+"\n")
        log_f.write("records:" + str(records)+"\n\n\n")

        df.loc[i, "IQ_llama_prob"] = mean
        df.loc[i, "IQ_llama_ans_list"] = str(records)
    
    log_f.write(f"\nDone at {time.ctime()}\n\n")

    df.to_csv(read_path, index=False)
    print(f"Wrote {counter:,} entries to {LOG_PATH} in {time.time() - start:.2f}s")    
        

def main_Q(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    read_path: str = None,
    WRITE_JSON_PATH: str = None,
    LOG_PATH: str = None,
    start_index: int = None,
    how_many: int = None, #-1 for all
):
    assert start_index is not None
    assert how_many is not None
    generator = gen.Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size)
        
    title_list = open(read_path,"r").readlines()
    res = []
    if start_index > 0:
        try:
            res = json.load(open(WRITE_JSON_PATH,"r"))
        except:
            print("No file found at {}".format(WRITE_JSON_PATH),"starting from scratch")
    counter = 0
    for i,entry in enumerate(title_list[start_index:]):
        if how_many != -1 and counter >= how_many:
            break
        counter += 1
        title = entry.strip()
        print("Processing title: {} index: {}".format(title,i))
        log_results = reference_query(generator,prompt_Q,title,max_gen_len,top_p,temperature,LOG_PATH)
        res.append(log_results)
            #Writing at every 20th iteration in case API crashes
        if (i+1)%20 == 0:
            json.dump(res,open(WRITE_JSON_PATH,"w"),indent=2,ensure_ascii=False)

    json.dump(res,open(WRITE_JSON_PATH,"w"),indent=2,ensure_ascii=False)
    convert_to_csv(WRITE_JSON_PATH)

def convert_to_csv(WRITE_JSON_PATH):
    WRITE_DF_PATH = WRITE_JSON_PATH.replace(".json",".csv")
    concept_lis = json.load(open(WRITE_JSON_PATH,"r"))
    res = []
    for i,concept in enumerate(concept_lis):
        for j,gen_title in enumerate(concept["gen_list"]):
            dict = {}
            gen_title = process_search_query(gen_title)
            dict["gen_title"] = gen_title
            dict["concept"] = concept["title"]
            dict["model_answer_main_query"] = concept["model_answer_main_query"]            
            print(i,j,"done")
            res.append(dict)
    df = pd.DataFrame(res)
    df.to_csv(WRITE_DF_PATH,index=False)
    with open(WRITE_JSON_PATH,"w") as f:
        json.dump(concept_lis,f,indent=2,ensure_ascii=False)
    add_bing_return(WRITE_DF_PATH)

def add_bing_return(WRITE_DF_PATH,n_threads=100):
    df = pd.read_csv(WRITE_DF_PATH)
    with multiprocessing.Pool(n_threads) as pool:
        results = pool.starmap(query_bing_return_mkt, [(row["gen_title"],) for i,row in df.iterrows()])
    print("DONE!!!")
    print(results[0][0])
    for i,row in df.iterrows():
        df.loc[i,"bing_return"] = results[i][0]
        df.loc[i,"bing_return_results"] = str(results[i][1])
    df.to_csv(WRITE_DF_PATH,index=False)
    print(df["bing_return"].value_counts())


def main(gen_type : str =None,
         ckpt_dir: str = None,
    tokenizer_path: str =None,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
    read_path: str = None,
    WRITE_JSON_PATH: str = None,
    LOG_PATH: str = None,
    start_index: int = None,
    num_gen: int =  None,
    how_many: int = None,
    dq_type: int = None):

    if gen_type == "Q":
        main_Q(ckpt_dir,tokenizer_path,temperature,top_p,max_seq_len,max_batch_size,max_gen_len,read_path,WRITE_JSON_PATH,LOG_PATH,start_index,how_many)
    elif gen_type == "IQ":
        main_IQ(ckpt_dir,tokenizer_path,temperature,num_gen,top_p,max_seq_len,max_batch_size,max_gen_len,read_path,LOG_PATH,start_index,how_many)
        df_extract_authors(read_path)
        main_CC(ckpt_dir,tokenizer_path,read_path,LOG_PATH,start_index,how_many)
    elif gen_type == "DQ":
        main_DQ(ckpt_dir,tokenizer_path,num_gen,temperature,top_p,max_seq_len,max_batch_size,max_gen_len,read_path,LOG_PATH,start_index,how_many,dq_type)
        correct_sample_dq(read_path)
        
        


if __name__ == "__main__":
    
    fire.Fire(main)
    
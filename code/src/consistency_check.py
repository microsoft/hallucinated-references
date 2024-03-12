import json
import tqdm as tqdm
import sys
from openai_query import chat, davinci_run
import pandas as pd
import time
import logging
import os
import argparse
import multiprocessing
from typing import List
import re
import shutil


PROMPT = """Below are what should be two lists of authors. On a scale of 0-100%, how much overlap is there in the author names (ignore minor variations such as middle initials or accents)? Answer with a number between 0 and 100. Also, provide a justification. Note: if either of them is not a list of authors, output 0. Output format should be ANS: <ans> JUSTIFICATION: <justification>.
1. <NAME_LIST1>
2. <NAME_LIST2>"""


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


def consistency_pair_check(list1: str, list2: str, model, temperature, max_chars=100, max_tokens=50):
    list1 = str(list1).strip()[:max_chars].replace("\n", " ")
    list2 = str(list2).strip()[:max_chars].replace("\n", " ")
    prompt = PROMPT.replace("<NAME_LIST1>", list1).replace("<NAME_LIST2>", list2)
    messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt}]
    # print("messages", messages)
    if model == "gpt-4" or model == "gpt-3.5-turbo" or model == "gpt-35-turbo":   
        ans = chat(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=1)[0]
    else:
        ans = davinci_run(model,prompt, temperature=temperature, max_tokens=max_tokens, n=1)[0]

    return (get_agreement_frac(ans), ans)


def consistency_check(auth_lists: List[str], model, temperature):
    n = len(auth_lists)
    assert n >= 2
    records = []
    fracs = []

    for i in range(n):
        for j in range(i):
            # print(f"Checking agreement between {i} and {j}")
            frac, ans = consistency_pair_check(auth_lists[i], auth_lists[j], model=model, temperature=temperature)
            records.append(ans)
            fracs.append(frac)
    mean = sum(fracs) / len(fracs)
    for a in auth_lists:
        print("    ", a)
    print(mean)
    print()
    return mean, records


def consistency_check_parallel(auth_lists_list: List[List[str]], model, temperature, n_threads):
    with multiprocessing.Pool(n_threads) as pool:
        results = pool.starmap(consistency_check, [(auth_lists, model, temperature) for auth_lists in auth_lists_list])
    return results


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    # add required filename positional argument
    parser.add_argument("--model", "-m", type=str, help="which to start at", required=True)
    parser.add_argument("--start", "-s", type=int, default=0, help="which to start at")
    parser.add_argument("--num", "-n", type=int, default=-1, help="How many to do, -1 means all")  
    parser.add_argument("--threads", "-t", type=int, default=1, help="How many threads")  
    parser.add_argument("--out_file", "-o", type=str,default="", help="Which .csv file to read/write to")
    parser.add_argument("--log", "-l", type=str, default="consistency_check.log", help="Which log file to write to")
    args = parser.parse_args()
    log_dir = "logs_1000_acm_"+args.model
    WRITE_DF_PATH = "logs/{}/{}_acm.csv".format(log_dir,args.model) 

    if args.out_file != "":
        assert not os.path.exists(args.out_file), "Output file already exists"
        # copy file from WRITE_DF_PATH to args.out_file
        shutil.copyfile(WRITE_DF_PATH, args.out_file)
        WRITE_DF_PATH = args.out_file

    df = pd.read_csv(WRITE_DF_PATH)

    with open(args.log, "a") as f:
        f.write(f"Starting at {time.ctime()} with arguments {args}\n")
    entries = {}

    for i, row in df.iterrows():
        if i < args.start:
            continue
        if args.num != -1 and len(entries) >= args.num:
            break
        entries[i] = [df.loc[i, f"model_ans_{j}"] for j in "123"]
    
    print(f"Running {args.model} on {len(entries):,} entries using {args.threads} threads")

    checks = consistency_check_parallel(list(entries.values()), args.model, 0.0, args.threads)

    with open(args.log, "a") as f:
        json.dump(checks, f, indent=4)
        f.write(f"\nDone at {time.ctime()}\n\n")

    for (frac, records), i in zip(checks, entries):
        df.loc[i, "neural_ans1_prob"] = frac
        df.loc[i, "neural_ans1_list"] = str(records)

    df.to_csv(WRITE_DF_PATH, index=False)
    print(f"Wrote {len(entries):,} entries to {WRITE_DF_PATH} in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()


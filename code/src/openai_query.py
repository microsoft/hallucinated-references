import openai
import time
import openai_key
import json
import subprocess

#openai.api_key = openai_key.key

def chat_completion_wrapper(model, messages, temperature, n, **args): 
    if openai.api_type == "azure":
        model = model.replace("3.5", "35")
        #print(f"Using azure api engine={model}, remove this print statement")
        return openai.ChatCompletion.create(engine=model,messages=messages,temperature=temperature,n=n)
    else:
        #print("Not using azure, remove this print statement")
        return openai.ChatCompletion.create(model=model,messages=messages,temperature=temperature,n=n)


def create_query(model,messages,temperature,n=1):
    openai.api_key = openai_key.key
    run_attempts = 20
    time_sleep = 1
    while run_attempts > 0:
        try:
            completion = chat_completion_wrapper(model, messages, temperature, n=n)
        except openai.error.RateLimitError:
            print("reading again, sleeping for", time_sleep)
            time.sleep(time_sleep)
            run_attempts -= 1
            time_sleep = time_sleep * 2
            if run_attempts == 0:
                    print("exiting")
                    exit()
            continue
        except Exception as e:
            print("other exception", e)
            time.sleep(time_sleep)
            run_attempts -= 1
            time_sleep = time_sleep * 2
            if run_attempts == 0:
                    print("exiting")
                    exit()
            continue
        
        break
    return completion

def chat(model,messages,temperature,n=1,max_tokens=None):
    """Only returns actual """
    openai.api_key = openai_key.key
    run_attempts = 200
    time_sleep = 0.03
    batch_size = n
    ans = []
    while len(ans) < n and run_attempts > 0:
        batch_size = min(batch_size, n - len(ans))
        try:
            completion = chat_completion_wrapper(model,messages=messages,temperature=temperature,n=batch_size,max_tokens=max_tokens)
            for choice in completion.choices:
                if "content" in choice.message:
                    ans.append(choice.message['content'].encode().decode('utf-8'))
                else:
                    print("content filtered!")
                    ans.append("<CONTENT FILTERED>")
        except openai.error.RateLimitError:
            print("reading again, sleeping for", time_sleep)
            time.sleep(time_sleep)
            run_attempts -= 1
            time_sleep = time_sleep * 1.2
            batch_size = max(1, batch_size // 2)
            print(f"Sleep time is now {time_sleep:.1f} batch size is now {batch_size}")
            if run_attempts == 0:
                    print("exiting")
                    exit()
            continue
        except Exception as e:
            print("other exception", e)
            time.sleep(time_sleep)
            run_attempts -= 1
            time_sleep = time_sleep * 2
            if run_attempts == 0:
                    print("exiting")
                    exit()
            continue
    return ans


def davinci_run(model,prompt,temperature,n=1,max_tokens=1000):
    query = {'model': model,'temperature':temperature,'max_tokens':max_tokens,'n':n, "prompt": prompt}
    run_attempts = 40
    time_sleep = 0.2
    while run_attempts > 0:
        try:
            output = subprocess.run(["curl", "--silent", 
                "https://api.openai.com/v1/completions", 
                "-H", "Content-Type: application/json", 
                "-H", f"Authorization: Bearer {openai_key.key}", 
                "-d", json.dumps(query)
                ], 
                capture_output=True
                )
            #print(output)
            output = output.stdout.decode("utf-8")
            output = json.loads(output)
            #print(output)
            res = []
            for choice in output["choices"]:
                res.append(choice["text"])
            break
        except openai.error.RateLimitError:
            print("reading again, sleeping for", time_sleep)
            time.sleep(time_sleep)
            run_attempts -= 1
            time_sleep = time_sleep * 1.2
            if run_attempts == 0:
                    print("exiting")
                    exit()
            continue
        except Exception as e:
            print("other exception", e)
            time.sleep(time_sleep)
            run_attempts -= 1
            time_sleep = time_sleep * 2
            if run_attempts == 0:
                    res = ["" for i in range(n)]
                    print("no. of attempts failed")
                    break
                    #print("exiting")
                    #exit()
            else:
                continue
        
        
    return res
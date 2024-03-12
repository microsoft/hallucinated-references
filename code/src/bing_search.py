import bing_key
import json
import string
import requests

subscription_key = bing_key.key

endpoint = "https://api.bing.microsoft.com/v7.0/search"

def process_string(s):
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.replace("“","")
    s = s.replace("”","")
    s = s.replace("’","")
    s = s.replace("‘","")
    s = s.lower()
    s = ''.join(s.split())
    return s

def process_search_query(s):
    s = s.strip()
    s = s.strip("\"")
    s = s.strip("“")
    s = s.strip("”")
    s = s.strip("’")
    s = s.strip("‘")
    s = s.strip("\"")
    s = s.strip(".")
    s = s.strip()
    return s


def query_bing_return_mkt(search_term,mkt=None):
    if "NOTHING" in search_term:
        return "NA",None
    search_term = process_search_query(search_term)

    #print(search_term)
    query = '"{}"'.format(search_term)
    #print(repr(query))
    # mkt = "ja-JP"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    if mkt is None:
        params = {"q": query, "textDecorations": True, "textFormat": "HTML", "count": 5}
    else:
        params = {"q": query, "textDecorations": True, "textFormat": "HTML", "count": 5,"mkt":mkt}
    
    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    #print(search_results)
    #market = search_results["queryContext"]["market"]
    #print("Market: {}\n".format(market))
    #print("================================================================")
    #print(search_results)
    #print("================================================================")
    if "webPages" in search_results:
        if search_results["webPages"]["totalEstimatedMatches"] > 0:
            return True,search_results
        else:
            return False,search_results
    else:
        return False,search_results


## DATASET OVERVIEW  

### BASICS: CONTACT, DISTRIBUTION, ACCESS

>Dataset Name  

Hallucinating References 
- References generated from 6 llms (`GPT-4`, `GPT-3.5-turbo`, `Text-davinci-003`, `Llama-2-7b-chat`, `Llama-2-13b-chat`, `Llama-2-70b-chat`) 
- expert-annotation-study (manual annotations of a random sample of the above dataset sample by the authors of the work) 


>Dataset version number or date  

first version, March 12, 2024 

>Dataset owner/manager contact information

- Ayush Agrawal, t-agrawalay@microsoft.com 
- Lester Mackey, lmackey@microsoft.com 
 

>Who can access this dataset?

Dataset will be open-sourced. Hence, anyone can access it. 

>How can the dataset be accessed?  

From the open-sourced github repo: https://github.com/microsoft/hallucinating-references

### DATASET CONTENTS  

> What are the contents of this dataset? 

- References generated 
    - `gen_title`: paper title generated from the LLM 
    - `title`: concept for which the title was generated 
    - `model_answer_main_query`: all the titles (total 5) for the particular concept generated from the LLM. 
    - `bing_return`: bing search label 
    - `model_ans1-3`: three times Sampled author names for the title from LLM 
    - `neural_ans1-3_list`: consistency scores as provided by the LLM 
    - `neural_ans1-3_prob`: probability calculated from the scores 

- Expert Annotation Study: 

    - `gen_title`: paper title sampled from the above dataset 
    - `search_url`: google-search url  
    - `label`: label provided by the expert annotator (one of the authors) 

>How many items are in the dataset?  

- References generated: 1000 items per model (total 6 models) 

- Expert Annotation study: 100 items per Annotator (total 5 annotators) 

>What data is available about each item?  

The data is collected from the LMs and contains paper titles and authors generated from the LM only with consistency scores as mentioned above in the datapoints fields. 

>For static datasets: What timeframe does the dataset cover?  

LM generations and Bing labels generated during August 2023 and human annotations collected during September 2023. 

### INTENDED & INAPPROPRIATE USES  

>What are the intended purposes for this dataset?  

Hallucination Research (hallucination detection and mitigation) 

>What are some tasks/purposes that this dataset is not appropriate for?  

One should only use this dataset as a representative for LMs hallucinations research and not rely on this dataset for any factual information. We do not propose to replace human verification and do not provide any guarantees on 100% hallucination detection.  We also do not provide guarantees that our Bing search labelling is 100% accurate.  

## DETAILS

### DATA COLLECTION PROCEDURES  

>How was the data collected?   

Datasets were collected from the language models (OpenAI and opensource) using their APIs. Labelling was carried out using Bing Search API. 

>Who collected the data?  

Authors of the work collected the entire dataset. 

>Describe considerations taken for responsible and ethical data collection. 

The dataset is collected entirely from the LMs, the Bing API, and the authors. In particular, the annotation study is carried out entirely by the authors. We advise users to be extremely careful while using model’s output references. 

>Describe procedures and include language used for getting explicit consent for data collection and use, and/or revoking consent (e.g., for future uses or for certain uses). If explicit consent was not secured, describe procedures and include language used for notifying people about data collection and use.  

English was used for instructing all the authors for annotations. All the prompts for the LMs were in English. 

### REPRESENTATIVENESS  

>How representative is this dataset? What population(s), contexts is it representative of?   

The LM generations are representative of the following language models collected during August 2023 when queried for book and paper titles on one of 200 computer science concepts defined by the Association for Computing Machinery (ACM): GPT-4, GPT-3.5-turbo, Text-davinci-003, Llama-2-7b-chat, Llama-2-13b-chat and Llama-2-70b-chat. The Bing labels are representative of the Bing API search results during August 2023. The human annotations are representative of the authors, all of whom are computer science researchers. 

>How was representativeness ensured or validated?  

We do not provide any validation for representativeness, but repeated independent querying of each model was designed to ensure representativeness. 

>What are known limits to this dataset’s representativeness?  

This dataset only represents computer science references (books and papers) related to 200 ACM concepts. In addition, the human annotations are only representative of the paper’s authors. 

>What demographic groups are identified in the dataset, if any?  

None 

>How were these demographic groups identified?  

NA 

>What is the breakdown of the dataset across demographic groups? Consider also reporting intersectional groups (e.g., race x gender) and including proportions, counts, means or other relevant summary statistics.  

NA 

### DATA QUALITY  

>Is there any missing information in the dataset? If yes, please explain what information is missing and why (e.g., some people did not report their gender).  

No 

>What errors, sources of noise, or redundancies are important for dataset users to be aware of?  

We do not provide guarantees for Bing search to be 100% accurate while labelling the titles. 

>What data might be out of date or no longer available (e.g., broken links in old tweets)? 

NA 

>How was the data validated/verified?  

An expert annotation study was conducted for 100 randomly selected sample points in the dataset. This study will also be open-sourced. 

>What are potential validity issues a user of this dataset needs to be aware of (e.g., survey answers might not be truthful, age was guessed by a model and might be incorrect, GPA was used to quantify intelligence)?  

The language model responses were generated by a model, and hence not all generated reference titles are grounded in reality.  In addition, the human annotators need not agree on the groundedness of each generated reference. 

>What are other potential data quality issues a user of this dataset needs to be aware of?  

Some model generations did not follow the instruction template entirely; for example, authors were sometimes wrongly positioned. However, these errors had little impact on the study.  

### PRE-PROCESSING, CLEANING, AND LABELING  

>What pre-processing, cleaning, and/or labeling was done on this dataset?  

Include information such as: how labels were obtained, treatment of missing values, grouping data into categories (e.g., was gender treated as a binary variable?), dropping data points.  

>Who did the pre-processing, cleaning, and/or labeling (e.g., were crowd workers involved in labeling?)  

Basic preprocessing was carried out where we extracted authors and individual titles from the LM outputs generated. Labelling was done using Bing Search. All this was done by the authors. No crowd workers were involved. 


>Provide a link to the code used to preprocess/clean/label the data, if available.   
 
https://github.com/microsoft/hallucinating-references/blob/main/code/src/generate_references.py

>If there are any recommended data splits (e.g., training, development/validation, testing), please explain.  

NA 

### PRIVACY  

>What are potential data confidentiality issues a user of this dataset needs to be aware of? How might a dataset user protect data confidentiality?  

Users should not rely on this dataset for any information. Apart from this, we do not see any vulnerabilities regarding confidentiality. 

>Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?  

One can identify persons using the author names of the paper titles if they exist. 

>Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals race, sexual orientation, age, ethnicity, disability status, political orientation, religious beliefs, union memberships; location; financial or health data; biometric or genetic data; criminal history)?  

We do not see any data to be sensitive. 

### ADDITIONAL DETAILS ON DISTRIBUTION & ACCESS

>How can dataset users receive information if this dataset is updated (e.g., corrections, additions, removals)?   

From the github page. 

>For static datasets: What will happen to older versions of the dataset? Will they continue to be maintained?  

They will be retained as part of the previous commits. 

>For streaming datasets: If this dataset pulls telemetry data from other sources, please specify:  What sources? How frequently the dataset is refreshed? Who controls access to these sources? Whether access to these sources will remain available, and for how long?  Any applicable access restrictions to these sources including licenses and fees? Any other available access points to these sources? Any relevant information about versioning?  

NA 

>Are there any other ways in which these sources might affect this dataset that a dataset user needs to be aware of?  

NA 

>If this dataset links to data from other sources (e.g., this dataset includes links to content such as social media posts or, news articles, but not the actual content), please specify:  

- What sources  
Google scholar, arXiv etc.

- Whether access to these sources will remain available, and for how long  
Depends on the policies of the above mentioned sources.

- Who controls access to these sources  
The respective sources.

- Any applicable access restrictions to these sources including licenses and fees  

NA 

>For static datasets: If an official archival version of the complete dataset exists (i.e., including the content as it was at the time the dataset was created), where it can be accessed  

On GitHub 

>Are there any other ways in which these sources might affect this dataset that a dataset user needs to be aware of?  

No 

>Describe any applicable intellectual property (IP) licenses, copyright, fees, terms of use, export controls, or other regulatory restrictions that apply to this dataset or individual data points.  These might include access restrictions related to data subjects’ consenting or being notified of data collection and use, as well as revoking consent.  Provide links to or copies of any such applicable terms.  

NA 






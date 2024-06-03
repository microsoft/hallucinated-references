### Summary 

Large language models (LLMs) are famous for generating factually incorrect books and papers titles. It becomes imperative to analyze this phenomenon known as hallucination in depth. We propose simple question answering strategies that aid in hallucinated reference titles detection by the LLMs themselves without relying on any external resources and model’s probability distribution. Our study provides an interesting insight into language models hallucination that might be mitigated by carefully designed decoding procedures. 

### Evaluation and Metrics 

We evaluated our approach on both huggingface and OpenAI models: Llama-2-chat (7B, 13B, 70B), GPT-3.5-turbo, Text-davinci-003 and GPT-4. We used exact match using Bing search API for labelling the generated references as Hallucinated (H) or Grounded (G) otherwise. We chose two standard metrics for evaluation: 

Receiver Operating Characteristic (ROC) Curves: Since each of our querying strategies outputs a real-valued score, one can trade off accuracy on G (i.e., how often truly grounded references are labeled G) and H (how often truly hallucinated references are labeled H) by thresholding the score to form a G or H classification. We visualize this trade-off using a standard receiver operating characteristic (ROC) curve and summarize overall detection performance using the area under the ROC curve (AUC). 

False Discovery Rate (FDR) Curves: Each groundedness classifier can also be used as a filter to generate a list of likely grounded references for a literature review based on the raw generations of an LM.  The two primary quantities of interest to a user of this filter would be the fraction of references preserved (more references provide a more comprehensive review) and the fraction of preserved references which are hallucinations. We show how these two quantities can be traded off using false discovery rate (FDR) curves. As one varies the threshold of G/H classification and returns only those references classified as grounded, the FDR captures the fraction of references produced which are hallucinations. 

### Intended Uses 

Our study aims to provide insights into hallucinated references detection. Users could use our approaches to detect hallucinated references or build upon our methods to generalize to other kinds of hallucinations. Additionally, our method could be used as a guiding light for development of better decoding techniques for hallucination mitigation. We do not propose to replace human verification and do not provide any guarantees on 100% hallucination detection.  

### Limitations 

There are several limitations of this work: 

1) Inaccessible training data: We consider web as a contending proxy for the models’ training data. However, we cannot conclude what is truly grounded versus hallucination since we do not have access to the training data.  

2) Hallucination spectrum: The notion of hallucination is not entirely black and white as considered in this work and in prior works. For example, a generated reference that is a substring or superstring of an existing title is hard to classify with the binary scheme.  

3) Prompt sensitivity: LMs are notoriously sensitive to prompt wording. Thus, some of our findings comparing direct and indirect queries may be sensitive to the specific wording in the prompt. 

4) Domain-specific reference bias: Since we use ACM Computing Classification System for our topics, the results are biased towards computer science references, though it would be straightforward to re-run the procedure on any given list of topics.  

5) Gender and racial biases: LMs have been shown to exhibit gender and racial biases which may be reflected in our procedure–in particular: our procedure may not recognize certain names as likely authors, or it may perform worse at matching names of people in certain racial groups where there is less variability in names. Since our work compares LMs and hallucination estimation procedures, the risk is lower compared to a system that might be deployed using our procedures to reduce hallucination. Before deploying any such system, one should perform a more thorough examination of potential biases against sensitive groups and accuracy across different research areas. 

### Effective and Responsible Use 

The users should not completely rely on our method for carrying out literature reviews etc. Instead, they should be extremely careful and use LLMs with caution while using model’s output references. We hope the users will gain from our hallucination study and use it in order to build safe and reliable AI systems. 
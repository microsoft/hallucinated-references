### Setup
- `python==3.8`
- `requests`
- `openai`
- `pandas>=1.4.4`
- `numpy>=1.23.0`
- download `auc_delong_xu.py` from [RaulSanchezVazquez/roc_curve_with_confidence_intervals](https://github.com/RaulSanchezVazquez/roc_curve_with_confidence_intervals)


### Running the experiments

#### For openai models
- create files `openai_key.py` and `bing_key.py` and put your respective API keys there as `key=<YOUR-KEY>` inside `src` folder.
- `cd src`
- `python generate_references.py -m "gpt-4" -s 0 -n -1` #runs to generate 1000 titles, creates `logs/logs_1000_acm_gpt-4` folder and creates `gpt-4_acm.csv` file containing the titles and bing search data

- `python cross_question.py -m "gpt-4" -s 0 -n -1` #questions author names for each title, loads the gpt-4_acm.csv and adds data to it. (IQ, step 1)
- `python consistency_check.py -m "gpt-4" -s 0 -n -1` #runs the overlapping procedure to calculate the scores. Adds data to the gpt-4_acm.csv file (IQ, step 2)
- `python neural_agreement.py -m "gpt-4" -s 0 -n -1` #running DQ1,DQ2,DQ3 methods, data is added to gpt-4_acm.csv
- `-m` could be set to the desired model.

#### For LlamaChat series
- clone the repo [facebookresearch/llama](https://github.com/facebookresearch/llama) and follow the instructions to get the models access.

- put the path of the llama repo in `llama_run.py` file at `<llama repo cloned path>`.
- torchrun --nproc_per_node 1/2/8 based on 7B, 13B, 70B models respectively as per the instructions in the repo is used to run the models.

- `llama_run.py --gen_type --ckpt_dir --tokenizer_path --temperature --max_gen_len --read_path --WRITE_JSON_PATH --LOG_PATH --start_index --num_gen --how_many --dq_type` command is run with torchrun.

- `gen_type` is either "Q" for generating references, "IQ","DQ" for generating answers.
- `ckpt_dir` is the path to the model checkpoint.
- `tokenizer_path` is the path to the tokenizer.
- `temperature` is the temperature for sampling.
- `max_gen_len` is the maximum length of the generated text.
- `read_path` is the path to the file containing the acm topics and for the IQ and DQ, path containing the csv files
- `WRITE_JSON_PATH` is the path to the folder where the generated json files are stored of the titles
- `LOG_PATH` is the path to the folder where the logs are stored
- `start_index` is the starting index of the titles to be generated
- `num_gen` is the number of answers to be sampled
- `how_many` is the number of titles to be generated
- `dq_type` is the type of DQ method to be run, either "1","2","3"


### Metrics

- `code_and_data/metrics.ipynb` notebook could be run setting the `PATH` variable to the correct csv file to get the ROC and FDR curves.

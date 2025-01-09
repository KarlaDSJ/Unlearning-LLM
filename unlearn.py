from transformers import AutoModelForCausalLM, AutoTokenizer
hf_token = "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"   #token

def unlearn(path_forget, 
            path_retain, 
            path_model, 
            path_checkpoints):
    '''
    Unlearning sensitive content (forget dataset) from Large Language Models

    Args:
        path_forget (str): Path for the private forget dataset (jsonl/parquet files) 
        path_retain (str): Path for the private retain dataset.
        path_model (str): Path to the fine tuned model path (includes the tokenizer).
        path_checkpoints (str): Path to the output directory to store the unlearned checkpoints.
    '''
    pass

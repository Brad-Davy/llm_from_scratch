from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
local_path = "./tinyllama-local"

# Download and save to local path
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(local_path)

model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(local_path)


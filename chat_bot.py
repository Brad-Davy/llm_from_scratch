from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pyfiglet

print(pyfiglet.figlet_format("OCF Chat Bot", font="slant"))

def generate_the_prompt(user_prompt: str) -> str:
    return f"### Human: {user_prompt} \n ### Assistant whos name is OCF chatbot:"

def clean_output(llm_output: str) -> str:
    print(f'-->{llm_output.splitlines()[1].split(":")[1]}')

local_path = "./tinyllama-local"

tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForCausalLM.from_pretrained(
    local_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_output(prompt: str = "Hello, what is your name and can you describe your purpose!") -> str:
    inputs = tokenizer(generate_the_prompt(prompt), return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=60,
            do_sample=False,            # <== Deterministic
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and strip prompt from response
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clean_output(decoded)



generate_output()





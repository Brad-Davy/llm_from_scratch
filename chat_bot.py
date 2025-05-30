from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pyfiglet

print(pyfiglet.figlet_format("OCF Chat Bot", font="slant"))

def generate_the_prompt(user_prompt: str) -> str:
    return f"### Human: {user_prompt} \n ### Assistant whos name is OCF chatbot:"

def clean_output(llm_output: str) -> str:
    cleaned_text = llm_output.splitlines()[1].split(":")[1].split('#')[0]
    print('\n')
    print(f'\033[31m (OCF Chat Bot) --> \033[0m {cleaned_text}')
    print('\n')

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
run = True  # initialize loop flag

while run:
    user_input = input("\033[32m (You) --> \033[0m")  # prompt the user

    if user_input.lower() == 'done':
        run = False  # fix: use assignment, not comparison
    else:
        generate_output(user_input)


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pyfiglet

print(pyfiglet.figlet_format("OCF Chat Bot", font="slant"))

def generate_the_prompt(user_prompt: str) -> str:
    return f"You are OCF chatbot, a helpful and concise assistant. Don't engage in dialouge, just answer the user. \nUser: {user_prompt}\nOCF chatbot:"

def clean_output(llm_output: str) -> str:
    
    raw_string = str(llm_output)
    response = raw_string.split("User:")[1].split('chatbot:')[1]
    print('\n')
    print(f'\033[31m (OCF Chat Bot) --> \033[0m {response}')
    print('\n')
    return response

local_path = "./tinyllama-local"

tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForCausalLM.from_pretrained(
    local_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_output(model_memory: list, prompt: str = "Hello, what is your name and can you describe your purpose!") -> str:
    
    full_prompt = ''.join(model_memory) + '\nUser: {user_prompt}\nOCF chatbot:'
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=200,
            do_sample=False,            # <== Deterministic
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )


    # Decode and strip prompt from response
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if full_output.startswith(prompt):
        response = full_output[len(prompt):].strip()
    else:
        response = full_output.strip()
    

    output = clean_output(response)
    return output

run = True  # initialize loop flag
memory = []
while run:
    user_input = input("\033[32m (You) --> \033[0m")  # prompt the user
    memory.append(f'User: {user_input}')
    if user_input.lower() == 'done' or user_input.lower() == 'exit':
        run = False  # fix: use assignment, not comparison
    else:
        output = generate_output(memory, user_input)
        memory.append(f'OCF chatbot:{output}')
        print(memory)


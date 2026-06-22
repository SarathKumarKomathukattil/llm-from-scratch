import torch
from torch import nn
import requests
import json
import time
from tqdm import tqdm
from pathlib import Path
import tiktoken
from torch.utils.data import Dataset,DataLoader
from basic_gpt2_model import GPTModel
from basic_gpt2_model import generate,text_to_token_ids,token_ids_to_text
from basic_gpt2_model import cal_loss_batch,cal_loss_loader,train_model_simple
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = tiktoken.get_encoding('gpt2')

NEW_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,      # 355M
    "n_heads": 16,        # 355M
    "n_layers": 24,       # 355M
    "drop_rate": 0.0,
    "qkv_bias": True
}

def download_and_load_file(file_path,url):
    if Path(file_path).exists():
        print("Skipping download, file already exists")
    else:
        request = requests.get(url,stream=True,timeout=60)
        with open(file_path,'wb') as file:
            for chunk in request.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
    with open(file_path,'r',encoding='utf-8') as file:
        data = json.load(file)
    return data


file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
)
data = download_and_load_file(file_path, url)



def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}")
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

train_portion = int(len(data) *0.85)
test_portion = int(len(data) *0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:test_portion+train_portion]
val_data = data[train_portion+test_portion:]

class InstructionDataset(Dataset):
    def __init__(self,data,tokenizer):
        super().__init__()
        self.data = data

        self.encoded_texts = []       
        for entry in self.data:
            instuction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instuction_plus_input + response_text

            self.encoded_texts.append(tokenizer.encode(full_text))      

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    
def custom_collate(batch,pad_token_id =50256,ignore_index=-100,allowed_max_length=None,device=device):
    batch_max_length = max(len(item)+1 for item in batch) #1 added so even the longest input has a padding
    inputs_list,targets_list = [],[]

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (new_item + [pad_token_id]*(batch_max_length-len(new_item)))

        inputs= torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_list.append(inputs)
        targets_list.append(targets)

    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)
    return inputs_tensor,targets_tensor


torch.manual_seed(123)
batch_size = 8
num_workers=0

train_dataset = InstructionDataset(train_data,tokenizer)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=custom_collate,
    drop_last=True
)
     
val_dataset = InstructionDataset(val_data,tokenizer)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=custom_collate,
    drop_last=False
)
     
test_dataset = InstructionDataset(test_data,tokenizer)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=custom_collate,
    drop_last=False
)    

model =GPTModel(NEW_CONFIG).to(device)
model.load_state_dict(torch.load('gpt2_355M.pth',map_location=device))


start_time = time.time()
num_epochs = 2
optimizer = torch.optim.AdamW(model.parameters(),lr =0.00005,weight_decay=0.1)

train_losses,val_losses,tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context=format_input(val_data[0]),
    tokenizer=tokenizer
)

end_time = time.time()
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")


def plot_losses(epochs_seen,tokens_seen,train_losses,val_losses):
    fig,ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(epochs_seen,train_losses,label="Training loss")
    ax1.plot(epochs_seen,val_losses,linestyle='dashed',label='Validation loss')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen,train_losses,alpha =0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.savefig('loss-plot.pdf')
    plt.show()


epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)



for entry in test_data[:3]:
    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx= text_to_token_ids(input_text,tokenizer).to(device),
        max_new_tokens=256,
        context_size=NEW_CONFIG["context_length"],
        eos_id=50256
    )

    generated_text = token_ids_to_text(token_ids,tokenizer)
    def response_format(generated_text):
        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:","")
            .strip()
        )
        return response_text

    print(input_text)
    print(f"\nCorrect response: {entry['output']}")
    print(f"\nModel response: {response_format(generated_text)}")
    print('............................................')


for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=NEW_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    test_data[i]["model_response"] = response_text

with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data,file,indent=4)
import requests
import zipfile
from pathlib import Path
import pandas as pd
import os
import time
import torch
from torch import nn
import tiktoken
from torch.utils.data import Dataset,DataLoader
from basic_gpt2_model import GPTModel
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



device = 'cuda' if torch.cuda.is_available() else 'cpu'
NEW_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,   # GPT-2's real context (not 256!)
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,         # off for inference
    "qkv_bias": True          # GPT-2 uses Q/K/V bias!
}


tokenizer = tiktoken.get_encoding('gpt2')

url = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
base_path = Path(r"C:\Users\ASUS\Documents\vscode-ml\LLMs-from-scratch-main\ch02\01_main-chapter-code")
zip_path = 'sms_spam_collection.zip'
extracted_path = 'sms_spam_collection'
data_file_path = base_path/extracted_path/"SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url,base_path,zip_path,extracted_path,data_file_path):
    if data_file_path.exists():
         print(f"{data_file_path} already exists. Skipping download and extraction.")
         return
    
    response = requests.get(url,stream=True,timeout=60)
    with open(base_path/zip_path,'wb') as file:
         for chunk in response.iter_content(chunk_size=8192):
              if chunk:
                   file.write(chunk)
    with zipfile.ZipFile(base_path/zip_path,'r') as zip_ref:
         zip_ref.extractall(base_path/extracted_path)
    os.rename(base_path/extracted_path/"SMSSpamCollection",base_path/extracted_path/"SMSSpamCollection.tsv")
    print(f"File downloaded and saved as SMSSpamCollection.tsv")
         
download_and_unzip_spam_data(url,base_path,zip_path,extracted_path,data_file_path)

df = pd.read_csv(base_path/extracted_path/"SMSSpamCollection.tsv",sep='\t',header=None,names=['Label','Text'])

def create_balanced_datset(df):
     num_spam = df[df['Label']=='spam'].shape[0]
     ham_subset = df[df['Label']=='ham'].sample(num_spam,random_state =123)
     balanced_df = pd.concat([ham_subset,df[df['Label']=='spam']])
     return balanced_df

balanced_df = create_balanced_datset(df)
map_dict = {'ham':0,'spam':1}
balanced_df['Label'] = balanced_df['Label'].map(map_dict)


def random_split(df,train_frac,validation_frac):
     df = df.sample(frac=1,random_state=123).reset_index(drop=True)
     train_end = int(len(df)*train_frac)
     validation_end = train_end + int((len(df)*validation_frac))

     train_df = df[:train_end]
     validation_df = df[train_end:validation_end]
     test_df = df[validation_end:]
     return train_df,validation_df,test_df
                 
train_df,validation_df,test_df=random_split(df=balanced_df,train_frac=0.7,validation_frac=0.1)

train_df.to_csv('train.csv',index=None)
validation_df.to_csv('validation.csv',index=None)
test_df.to_csv('test.csv',index=None)

class SpamDataset(Dataset):
    def __init__(self,csv_file,tokenizer,max_length = None,pad_token_id=50256):
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data['Text']]
        
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [encoded_text[:max_length] for encoded_text in self.encoded_texts]
        
        self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) 
                              for encoded_text in self.encoded_texts]


    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]['Label']
        return (torch.tensor(encoded,dtype=torch.long),torch.tensor(label,dtype=torch.long))

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if max_length < encoded_length:
                max_length = encoded_length
        return max_length

    
train_dataset = SpamDataset(
    csv_file='train.csv',
    tokenizer=tokenizer,
    max_length=None
)

validation_dataset = SpamDataset(
    csv_file='validation.csv',
    tokenizer=tokenizer,
    max_length=train_dataset.max_length
)

test_dataset = SpamDataset(
    csv_file='test.csv',
    tokenizer=tokenizer,
    max_length=train_dataset.max_length
)

num_workers = 0
batch_size = 8
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)

val_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)


model = GPTModel(NEW_CONFIG).to(device)
model.load_state_dict(torch.load('gpt2_124M.pth',map_location=device))

# Freezing
for param in model.parameters():
    param.requires_grad = False

torch.manual_seed(123)
model.out_head = nn.Linear(in_features=768, out_features=2, bias=False)

for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

model.to(device)

def cal_accuracy_loader(data_loader,model,device,num_batches = None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches,len(data_loader))
    for i,(input_batch,target_batch) in enumerate(data_loader):
        if i<num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:,-1,:]
            predicted_labels = torch.argmax(logits,dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions/num_examples

def cal_loss_batch(input_batch,target_batch,model,device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:,-1,:]
    loss = nn.functional.cross_entropy(logits,target_batch)
    return loss

def cal_loss_loader(data_loader,model,device,num_batches = None):
    total_loss = 0
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches,len(data_loader))
    for i,(input_batch,target_batch) in enumerate(data_loader):
        if i<num_batches:
            loss = cal_loss_batch(input_batch,target_batch,model,device)
            total_loss += loss.item()
        else:
            break
    return total_loss/num_batches

def train_classifier_simple(model, train_loader,val_loader, optimizer, device,
                            num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen,global_step = 0,-1
    for epoch in range(num_epochs):
        model.train()
        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()
            loss = cal_loss_batch(input_batch, target_batch,model,device)
            loss.backward()
            optimizer.step()
            examples_seen += 1
            global_step += 1
         
            # Periodic evaluation
            if global_step % eval_freq ==0:
                train_loss,val_loss = evaluate_model(model,train_loader,val_loader,device,eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")


                train_accuracy = cal_accuracy_loader(train_loader,model,device,num_batches=eval_iter)
                val_accuracy = cal_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
                print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
                print(f"Validation accuracy: {val_accuracy*100:.2f}%")
                train_accs.append(train_accuracy)
                val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def evaluate_model(model,train_loader,val_loader,device,eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = cal_loss_loader(train_loader,model,device,num_batches=eval_iter)
        val_loss = cal_loss_loader(val_loader,model,device,num_batches=eval_iter)
    model.train()
    return train_loss,val_loss


start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(),lr= 5e-5,weight_decay=0.1)
num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(model,train_loader,val_loader,optimizer,
                                                                                        device,num_epochs=num_epochs,eval_freq=50,eval_iter=5)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

import matplotlib.pyplot as plt
import torch


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation values against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Second x-axis for examples seen
    ax2 = ax1.twiny()                                # shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)   # invisible plot to align ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


# ---- plot the LOSS curve ----
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses, label="loss")

# ---- plot the ACCURACY curve ----
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")
   
train_accuracy = cal_accuracy_loader(train_loader, model, device)
val_accuracy = cal_accuracy_loader(val_loader, model, device)
test_accuracy = cal_accuracy_loader(test_loader, model, device)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

def classify_text(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare the input
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    input_ids = input_ids[:min(max_length, supported_context_length)]   # truncate if needed
    input_ids += [pad_token_id] * (max_length - len(input_ids))         # pad to max_length
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # add batch dim

    # Predict
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]      # last token's scores
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"


# ---- try it on a message ----
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print(classify_text(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))            












import requests
from pathlib import Path
import tiktoken
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


device = 'cuda' if torch.cuda.is_available() else 'cpu'


GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 256,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}
cfg = GPT_CONFIG_124M


file_path = Path('the-verdict.txt')
if not file_path.exists():
    request = requests.get('https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt')
    with open('the-verdict.txt','wb') as file:
        file.write(request.content)

# Data sampling with a sliding window
with open('the-verdict.txt','r',encoding='utf-8') as file:
    raw_text = file.read()

tokenizer = tiktoken.get_encoding('gpt2')
class GPTDatasetV1(Dataset):
    def __init__(self,txt,max_length,stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text=txt,allowed_special={'<|endoftext|>'})

        for i in range(0,len(token_ids) - max_length,stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length +1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]

def create_dataloader_v1(txt,max_length,batch_size,shuffle,num_workers,drop_last,stride):

    dataset = GPTDatasetV1(txt,max_length,stride)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=drop_last)
    return dataloader

# Train/Validation ratio
train_ratio = 0.90
split_idx = int(len(raw_text) * train_ratio)
train_data = raw_text[:split_idx]
val_data = raw_text[split_idx:]

train_loader = create_dataloader_v1(
    txt=train_data,
    max_length=cfg["context_length"],
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True,
    stride=cfg["context_length"]
)

val_loader = create_dataloader_v1(
    txt=val_data,
    max_length=cfg["context_length"],
    batch_size=2,
    shuffle=False,
    num_workers=0,
    drop_last=False,
    stride=cfg["context_length"]
)

class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,num_heads,context_length,dropout,qkv_bias):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out//num_heads

        self.W_query = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias = qkv_bias)

        self.out_proj = nn.Linear(d_out,d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',(torch.triu(torch.ones(context_length,context_length),diagonal=1)))

    def forward(self,x):
        b,num_tokens,d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b,num_tokens,self.num_heads,self.head_dim)
        keys = keys.view(b,num_tokens,self.num_heads,self.head_dim)
        values = values.view(b,num_tokens,self.num_heads,self.head_dim)

        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]
        attn_scores.masked_fill_(mask_bool,-torch.inf)

        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1,2)
        context_vec = context_vec.contiguous().view(b,num_tokens,self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


# LLM architecture
class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))


    def forward(self,x):
        mean = x.mean(dim=-1,keepdim =True)
        var = x.var(dim=-1,keepdim =True,unbiased = False)
        norm_x = (x-mean)/torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.shift
    
class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 *cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    def forward(self,x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in= cfg["emb_dim"],
            d_out= cfg["emb_dim"],
            num_heads= cfg["n_heads"],
            context_length= cfg["context_length"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg=cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self,x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x += shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x += shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"],cfg["vocab_size"],bias=False
        )

    def forward(self,in_idx):
        batch_size,seq_len = in_idx.shape
        tok_embs = self.tok_emb(in_idx)
        pos_embs = self.pos_emb(torch.arange(seq_len,device=in_idx.device))
        x = tok_embs + pos_embs
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
torch.manual_seed(123)   
model  = GPTModel(cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr = 0.0004,weight_decay = 0.1)

# Calculate loss
def cal_loss_batch(input_batch,target_batch,model,device):
    input_batch,target_batch = input_batch.to(device),target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())
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
        if i < num_batches:
            loss =  cal_loss_batch(input_batch,target_batch,model,device)
            total_loss += loss.item()
        else:
            break
    return total_loss/num_batches

def text_to_token_ids(text,tokenizer):
    encoded = tokenizer.encode(text,allowed_special = {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids,tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())    

def generate(model,idx,max_new_tokens,context_size,top_k = None,temperature = 0.0,eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]

        if top_k is not None:
           top_logits,top_pos =  torch.topk(logits,top_k)
           logits = torch.where(
               condition= logits < top_logits[:,-1],
               input = torch.tensor(float("-inf")),
               other = logits
           )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
        else:
            idx_next = torch.argmax(logits,dim=-1,keepdim=True)

        if eos_id is not None and idx_next.item() == eos_id:   # ← stop if end token
            break
        idx = torch.cat((idx,idx_next),dim=1)
    return idx


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses,val_losses,track_tokens_seen = [],[],[]
    tokens_seen,global_step = 0,-1
    for epoch in range(num_epochs):
        model.train()

        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()
            loss = cal_loss_batch(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            #Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss,val_loss = evaluate_model(model,train_loader,val_loader,device,eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch: {epoch+1} (Step {global_step:06d})| "
                      f"Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses,val_losses,track_tokens_seen

def evaluate_model(model,train_loader,val_loader,device,eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = cal_loss_loader(train_loader,model,device,num_batches=eval_iter)
        val_loss = cal_loss_loader(val_loader,model,device,num_batches=eval_iter)
    model.train()
    return train_loss,val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=15, context_size=context_size,top_k=25,temperature=1.4
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  
    model.train()


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



if __name__ == "__main__":
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)











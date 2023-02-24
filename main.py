import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters – not learned by model
batch_size = 16 # Sequences to process in parallel
block_size = 32 # Maxmimum context length for predictions
max_iters = 5000 # Iterations for transformer
eval_interval = 100 # Interval for evaluation of loss
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # Iterations for evaluation
n_embd = 64 # Embedding dimension for input
n_head = 4 # Number of heads in each multi-head attention block
n_layer = 4 # Number of self-attention layers
dropout = 0.0 # Avoid overfitting

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read() # Read in the data

chars = sorted(list(set(text))) # All possible characters in set.
vocab_size = len(chars) # Length of dictionary

# Creating a mapping from characters to integers (simple encoder / decoder)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # Encoder -> Take string, output list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # Decoder -> Take list of integers, output string

# Create train and testing splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 90% = training data, 10% = validation data
train_data = data[:n]
val_data = data[n:]

# Loading in data
def get_batch(split):
    # Generates a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data # Selects dataset to use
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # Adds to device being used
    return x, y

@torch.no_grad()
def estimate_loss():
    # Evaluates the loss between training and validation sets
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    # Creating a single head of self-attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input here is BTC, output is BTHS
        B,T,C = x.shape 
        k = self.key(x)
        q = self.query(x)
        # Computing attention scores -> affinities between different variables
        attention = q @ k.transpose(-2, -1) * C**-0.5 # Returns BTT
        attention = attention.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Masking future values
        attention = F.softmax(attention, dim=-1) # Softmaxing values to maximumize and normalize
        attention = self.dropout(attention)
        v = self.value(x)
        out = attention @ v # Generating final attention matrix
        return out

class MultiHeadAttention(nn.Module):
    # Creating multiple heads of self-attention in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)    

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # Concatenates output of multiple heads
        out = self.dropout(self.proj(out))
        return out
 
class FeedFoward(nn.Module):
    # Simple feed forward neural network with a non-linearity
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential( # Sequential stacks modules and applies them to input tensor
            nn.Linear(n_embd, 4 * n_embd), # Takes input tensor and produces output of 4x size
            nn.ReLU(), # Applies an element-wise nonlinearity
            nn.Linear(4 * n_embd, n_embd), # Reduces tensor to normal size
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # Creating an entire attention block
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head # Size of each head in our multi-head attention block
        self.sa = MultiHeadAttention(n_head, head_size) # Creating a new multi-head attention block
        self.ffwd = FeedFoward(n_embd) # Creating a new feed forward network
        self.ln1 = nn.LayerNorm(n_embd) # Layer of normalization to self-attention layer
        self.ln2 = nn.LayerNorm(n_embd) # Layer of normalization to feed forward network
    
    # Add component / skip connection
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    # Creating the language model
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # Create an embedding table for tokens
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # Create an embedding table for positional encoding
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # Creates multiple self-attention layers
        self.ln_f = nn.LayerNorm(n_embd) # Normalization layer
        self.lm_head = nn.Linear(n_embd, vocab_size) # Maps output to probability distribution

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # Embed input
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # Positional encoding
        x = tok_emb + pos_emb # Concatenating positional encoding
        x = self.blocks(x) # Passing through self-attention blocks 
        x = self.ln_f(x)
        logits = self.lm_head(x) # Generating logits -> probability distribution

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # Calculating loss
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Generates new words based on given context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # Crop IDX
            logits, loss = self(idx_cond) # Generate predictions
            logits = logits[:, -1, :]  # Focus on the last time step
            probs = F.softmax(logits, dim=-1) # Apply softmax to generate probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample from distribution
            idx = torch.cat((idx, idx_next), dim=1) # Add sampled index to generated text
        return idx

model = GPTLanguageModel()
print("Here we go!")
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # PyTorch optimizer

for iter in range(max_iters):
    # Evaluate loss every so often
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train') # Sample a batch of data
    logits, loss = model(xb, yb) # Evaluate the loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
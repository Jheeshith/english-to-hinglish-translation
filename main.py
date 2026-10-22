"""
Question 19: English-to-Hinglish Machine Translation
Dataset: Kaggle — English–Hinglish Machine Translation (eng_hinglish_translation.csv)
Tasks:
  1. Load dataset and preprocess paired English and Hinglish sentences.
  2. Tokenize both languages, create vocabularies, and pad sequences.
  3. Design and implement an LSTM-based Sequence-to-Sequence Encoder–Decoder model.
  4. Train model using teacher forcing and evaluate translation quality using BLEU.
  5. Demonstrate trained model by translating at least five unseen English sentences into Hinglish.
"""

import os
import re
import math
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"==================================================")
print(f" QUESTION 19: ENGLISH-TO-HINGLISH TRANSLATION     ")
print(f" Computing Device: {device}                       ")
print(f"==================================================")

# 1. Load Dataset & Preprocess
data_path = os.path.join(os.path.dirname(__file__), 'data', 'eng_hinglish_translation.csv')
df = pd.read_csv(data_path)

def clean_sentence(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

df['clean_eng'] = df['english'].apply(clean_sentence)
df['clean_hing'] = df['hinglish'].apply(clean_sentence)
print(f"Loaded {len(df)} parallel English-Hinglish translation pairs.")

# 2. Tokenize & Build Vocabularies
PAD_TOKEN = 0
UNK_TOKEN = 1
SOS_TOKEN = 2
EOS_TOKEN = 3

eng_vocab = {'<PAD>': PAD_TOKEN, '<UNK>': UNK_TOKEN, '<SOS>': SOS_TOKEN, '<EOS>': EOS_TOKEN}
hing_vocab = {'<PAD>': PAD_TOKEN, '<UNK>': UNK_TOKEN, '<SOS>': SOS_TOKEN, '<EOS>': EOS_TOKEN}

for s in df['clean_eng']:
    for w in s.split():
        if w not in eng_vocab:
            eng_vocab[w] = len(eng_vocab)

for s in df['clean_hing']:
    for w in s.split():
        if w not in hing_vocab:
            hing_vocab[w] = len(hing_vocab)

inv_hing_vocab = {idx: w for w, idx in hing_vocab.items()}
print(f"English Vocabulary Size:  {len(eng_vocab)} tokens")
print(f"Hinglish Vocabulary Size: {len(hing_vocab)} tokens")

MAX_ENG_LEN = 15
MAX_HING_LEN = 15

def encode_src(sentence, vocab, max_len):
    tokens = sentence.split()
    ids = [vocab.get(w, UNK_TOKEN) for w in tokens][:max_len - 1] + [EOS_TOKEN]
    if len(ids) < max_len:
        ids += [PAD_TOKEN] * (max_len - len(ids))
    return ids

def encode_tgt(sentence, vocab, max_len):
    tokens = sentence.split()
    ids = [SOS_TOKEN] + [vocab.get(w, UNK_TOKEN) for w in tokens][:max_len - 2] + [EOS_TOKEN]
    if len(ids) < max_len:
        ids += [PAD_TOKEN] * (max_len - len(ids))
    return ids

X_src = np.array([encode_src(s, eng_vocab, MAX_ENG_LEN) for s in df['clean_eng']])
Y_tgt = np.array([encode_tgt(s, hing_vocab, MAX_HING_LEN) for s in df['clean_hing']])

X_train, X_test, y_train, y_test, raw_eng_train, raw_eng_test, raw_hing_train, raw_hing_test = train_test_split(
    X_src, Y_tgt, df['english'].values, df['hinglish'].values, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Sequence-to-Sequence Encoder-Decoder Architecture
"""
Model Justification:
An LSTM-based Encoder-Decoder architecture is selected:
1. Machine translation maps variable-length source sequences (English) to variable-length target sequences (Hinglish).
2. The Encoder compresses the source syntactic and semantic information into context vectors (hidden and cell states).
3. The Decoder sequentially unfolds the context vector, generating target tokens using Teacher Forcing during training.
"""
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=128, hidden_dim=256):
        super(Seq2Seq, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, src, tgt, teacher_forcing_ratio=0.75):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.fc_out.out_features
        
        # Encoder forward pass
        src_emb = self.src_embedding(src)
        _, (hidden, cell) = self.encoder(src_emb)
        
        outputs = torch.zeros(batch_size, tgt_len - 1, tgt_vocab_size).to(device)
        dec_input = tgt[:, 0].unsqueeze(1) # Start with <SOS>
        
        for t in range(1, tgt_len):
            tgt_emb = self.tgt_embedding(dec_input)
            dec_out, (hidden, cell) = self.decoder(tgt_emb, (hidden, cell))
            prediction = self.fc_out(dec_out.squeeze(1)) # [batch_size, vocab_size]
            outputs[:, t - 1, :] = prediction
            
            # Teacher forcing
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            dec_input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
            
        return outputs

model = Seq2Seq(src_vocab_size=len(eng_vocab), tgt_vocab_size=len(hing_vocab)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# 4. Training Loop with Teacher Forcing
EPOCHS = 15
train_losses = []
print("\nTraining Seq2Seq Translation Model...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        output = model(bx, by, teacher_forcing_ratio=0.75)
        # by[:, 1:] is target without <SOS>
        loss = criterion(output.reshape(-1, output.size(-1)), by[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * bx.size(0)
    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)
    if epoch % 3 == 0 or epoch == 1:
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] - Cross-Entropy Loss: {epoch_loss:.4f}")

# 5. Translation Inference Function (Greedy Decoding)
def translate_sentence(model, sentence, max_len=15):
    model.eval()
    tokens = sentence.lower().split()
    enc_src = [eng_vocab.get(w, UNK_TOKEN) for w in tokens][:max_len - 1] + [EOS_TOKEN]
    src_tensor = torch.tensor([enc_src], dtype=torch.long).to(device)
    
    with torch.no_grad():
        src_emb = model.src_embedding(src_tensor)
        _, (hidden, cell) = model.encoder(src_emb)
        
        dec_input = torch.tensor([[SOS_TOKEN]], dtype=torch.long).to(device)
        translated_tokens = []
        
        for _ in range(max_len):
            tgt_emb = model.tgt_embedding(dec_input)
            dec_out, (hidden, cell) = model.decoder(tgt_emb, (hidden, cell))
            prediction = model.fc_out(dec_out.squeeze(1))
            top1 = prediction.argmax(1).item()
            
            if top1 == EOS_TOKEN:
                break
            if top1 not in (PAD_TOKEN, SOS_TOKEN):
                translated_tokens.append(inv_hing_vocab.get(top1, '<UNK>'))
            dec_input = torch.tensor([[top1]], dtype=torch.long).to(device)
            
    return " ".join(translated_tokens)

# 6. Evaluation using Corpus-level BLEU Metric
def calculate_bleu(reference, candidate, max_n=2):
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    if len(cand_tokens) == 0:
        return 0.0
    
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)])
        cand_ngrams = Counter([tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1)])
        overlap = sum((cand_ngrams & ref_ngrams).values())
        total = max(1, sum(cand_ngrams.values()))
        precisions.append(overlap / total)
    
    # Brevity penalty
    c = len(cand_tokens)
    r = len(ref_tokens)
    bp = 1.0 if c > r else math.exp(1 - r / max(1, c))
    score = bp * math.exp(sum(math.log(max(1e-4, p)) for p in precisions) / len(precisions))
    return score

bleu_scores = []
for i in range(min(100, len(raw_eng_test))):
    pred_trans = translate_sentence(model, raw_eng_test[i])
    bleu_scores.append(calculate_bleu(clean_sentence(raw_hing_test[i]), pred_trans))

avg_bleu = np.mean(bleu_scores)
print(f"\n================ Translation Quality Metric ================")
print(f"  Average BLEU-2 Score on Test Set: {avg_bleu * 100:.2f} %")
print(f"============================================================")

# 7. Visualization of Training Loss
plt.figure(figsize=(7, 4))
plt.plot(train_losses, color='darkmagenta', lw=2)
plt.title('Seq2Seq Translation Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), 'actual_vs_predicted.png')
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Saved training loss plot to: {plot_path}")

# 8. Demonstration on At Least Five Unseen English Sentences
unseen_sentences = [
    "How are you today?",
    "Where are you going right now?",
    "I am learning deep learning programming.",
    "What time will the meeting start?",
    "Please call me when you reach home."
]

print(f"\nDemonstration on 5 Unseen English Sentences into Hinglish:")
print("=" * 85)
for s in unseen_sentences:
    translation = translate_sentence(model, s)
    print(f"English Input:   \"{s}\"")
    print(f"Hinglish Output: \"{translation}\"")
    print("-" * 85)
print("============================================================\n")

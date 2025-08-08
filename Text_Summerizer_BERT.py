import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('/content/NewsDataset.csv', nrows = 21000, on_bad_lines='skip') # nrows = 5000,

df = df.astype('str')
df.head()

df.drop_duplicates(subset=['Text', 'Summary'], inplace=True)
df.dropna(axis=0, inplace=True)

df.shape

contractions_dict = {
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have",
}

def expand_contractions(text, contractions_dict):
    for contraction, expansion in contractions_dict.items():
        text = text.replace(contraction, expansion)
    return text

df['contrac'] = df['Text'].apply(lambda x: expand_contractions(x, contractions_dict))

import string
punch = string.punctuation
punch

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk_sw = set(stopwords.words('english'))

import re
def tweet_cleaner_with_swr(text):
    new_text = re.sub(r'https?://\S+|www\.\S+','',text.lower())
    new_text = re.sub(r"'s\b", " is", new_text)
    new_text = re.sub(r'\([^)]*\)','',new_text)
    new_text = re.sub(r'<.*?>',' ',new_text)
    new_text = re.sub('@[A-Za-z0-9]+',' ',new_text)
    new_text = re.sub(r"[^a-zA-Z]", " ", new_text)
    new_text = re.sub(r'\b(\w+)(?:\W+\1\b)+',r"\1", new_text)
    new_text = re.sub(r'(.)\1{2,}', r'\1\1', new_text)

    clean_txt = ''
    for word in new_text.split():
        if word not in nltk_sw and len(word)>2:
                clean_txt = clean_txt + word + ' '
    return clean_txt.strip()

df['contrac_sum'] = df['Summary'].apply(lambda x: expand_contractions(x, contractions_dict))

df['clean_sum'] = df['contrac_sum'].apply(tweet_cleaner_with_swr)

df['clean_text'] = df['Text'].apply(tweet_cleaner_with_swr)
for i in range(5):
    print('Orignal: ', df['Text'].iloc[i])
    print()
    print('Review: ', df['clean_text'].iloc[i])
    print()
    print('Summary: ', df['clean_sum'].iloc[i])
    print()
    print(30*"--")
    print()

df.drop_duplicates(subset=['Text', 'Summary'],keep='first', inplace=True)
df.replace('', np.nan, inplace=True)
df.dropna(axis=0, inplace=True)
df.shape

import matplotlib.pyplot as plt
import seaborn as sns
nltk.download('punkt')
df['text_len'] = df['clean_text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['sum_len'] = df['clean_sum'].apply(lambda x: len(nltk.word_tokenize(x)))

plt.figure(figsize=(16,5))
ax = sns.countplot(x='sum_len', data=df[(df['sum_len']>=0)], palette='mako')
plt.title('summary length more than 10 words', fontsize=20)
plt.yticks([])
ax.bar_label(ax.containers[0])
plt.ylabel('count')
plt.xlabel('')
plt.show()

plt.figure(figsize=(16,5))
ax = sns.countplot(x='text_len', data=df[(df['text_len']>=0) & (df['text_len']<=50)], palette='mako')
plt.title('text length more than 10 words', fontsize=20)
plt.yticks([])
ax.bar_label(ax.containers[0])
plt.ylabel('count')
plt.xlabel('')
plt.show()

df.info()

ds = df[(df['text_len']<=39) & (df['text_len']>=28)]
ds = ds[(df['sum_len']>=6) & (ds['sum_len']<=8)]

ds.describe()


from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(np.array(ds['clean_text']), np.array(ds['clean_sum']), test_size=0.1, random_state=42, shuffle=True)

print(y_tr[0])

bert = BertModel.from_pretrained('bert-base-uncased').to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenizer.bos_token = '[CLS]'
tokenizer.eos_token = '[SEP]'

bert.resize_token_embeddings(len(tokenizer))

text_len = [len(tokenizer.encode(t)) for t in x_tr]
print("Avg input length:", np.mean(text_len), "Max:", max(text_len), np.quantile(text_len, .8))

summ_len = [len(tokenizer.encode(t)) for t in y_tr]
print("Avg input length:", np.mean(summ_len), "Max:", max(summ_len), np.quantile(summ_len, .8))

plt.figure(figsize=(16,5))
plt.hist(text_len, bins=100)
plt.show()

plt.figure(figsize=(16,5))
plt.hist(summ_len)
plt.ylabel('count')
plt.xlabel('')
plt.show()

print(tokenizer.bos_token_id)
print(tokenizer.eos_token_id)

max_len_text = 44
max_len_summary = 11

tokens_train = tokenizer.batch_encode_plus(
    np.array(x_tr).tolist(),
    max_length=max_len_text, padding="max_length", truncation=True
)
tokens_val = tokenizer.batch_encode_plus(
    np.array(x_val).tolist(),
    max_length=max_len_text, padding="max_length", truncation=True
)

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])

print(tokenizer.special_tokens_map)

y_tokens_train = tokenizer.batch_encode_plus(
    np.array(y_tr).tolist(),
    max_length=max_len_summary+4, padding="max_length", truncation=True
)
y_tokens_val = tokenizer.batch_encode_plus(
    np.array(y_val).tolist(),
    max_length=max_len_summary+4, padding="max_length", truncation=True
)

y_train_seq = torch.tensor(y_tokens_train['input_ids'])
y_train_mask = torch.tensor(y_tokens_train['attention_mask'])
y_val_seq = torch.tensor(y_tokens_val['input_ids'])
y_val_mask = torch.tensor(y_tokens_val['attention_mask'])

from torch.utils.data import DataLoader, Dataset

class SummarizationDataset(Dataset):
    def __init__(self, input_ids, attention_mask, target):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.target = target

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx], self.target[idx])

train_dataset = SummarizationDataset(train_seq, train_mask, y_train_seq)
val_dataset = SummarizationDataset(val_seq, val_mask, y_val_seq)

BATCH_SIZE = 64
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# train_dataset

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        if encoder_outputs.dim() == 2:
            encoder_outputs = encoder_outputs.unsqueeze(1)

        seq_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention_weights = torch.softmax(self.v(energy).squeeze(2), dim=1)

        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights

class Encoder(nn.Module):
    def __init__(self, bert_model, hidden_dim):
        super(Encoder, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        encoder_out = self.fc(sequence_output)  # (batch_size, seq_len, hidden_dim)
        return encoder_out

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embed_dim, num_layers=2):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_emb = nn.Dropout(0.3)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim + embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.layer_norm = nn.LayerNorm(hidden_dim*2)
        self.dropout_out = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, decoder_input, hidden, cell, encoder_outputs):
        embedded = self.dropout_emb(self.embedding(decoder_input)).unsqueeze(1)

        context, _ = self.attention(hidden[-1], encoder_outputs)
        context = context.unsqueeze(1)

        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        combined = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)
        combined = self.layer_norm(combined)
        combined = self.dropout_out(combined)

        return self.fc(combined), hidden, cell

print(tokenizer.special_tokens_map)

i = np.random.randint(0, len(train_seq))
print(x_tr[i])
print(tokenizer.convert_ids_to_tokens(train_seq[i]))
print(y_tr[i])
print(tokenizer.convert_ids_to_tokens(y_train_seq[i]))
print(len(tokenizer.convert_ids_to_tokens(y_train_seq[i])))

import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, attention_mask, target, epoch, max_epochs):
        batch_size, target_len = target.shape
        outputs = torch.zeros(batch_size, target_len, self.decoder.fc.out_features).to(device)

        encoder_outputs = self.encoder(input_ids, attention_mask)

        pooled = encoder_outputs.mean(dim=1)
        hidden = torch.stack([pooled] * self.decoder.lstm.num_layers)
        cell = torch.zeros_like(hidden).to(device)

        teacher_forcing_ratio = max(0.9 - epoch*0.02, 0.3)

        decoder_input = target[:, 0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output

            use_teacher_forcing = t < target_len * teacher_forcing_ratio
            decoder_input = target[:, t] if use_teacher_forcing else output.argmax(-1)

            if not use_teacher_forcing and (decoder_input == tokenizer.convert_tokens_to_ids('[SEP]')).all(): # changed <eos>
                break

        return outputs

def evaluate_model(model, val_dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask, target in val_dataloader:
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            outputs = model(input_ids, attention_mask, target, epoch=0, max_epochs=1)
            loss = criterion(outputs.view(-1, VOCAB_SIZE), target.view(-1))
            total_loss += loss.item()
    return total_loss / len(val_dataloader)

len(tokenizer), tokenizer.vocab_size

for param in bert.parameters():
    param.requires_grad = False

import torch
from torch.optim import AdamW
import tqdm as tqdm

HIDDEN_DIM = 256
EMBED_DIM = 256+128
VOCAB_SIZE = len(tokenizer)

encoder = Encoder(bert, HIDDEN_DIM).to(device)
decoder = Decoder(VOCAB_SIZE, HIDDEN_DIM, EMBED_DIM).to(device)
model = Seq2Seq(encoder, decoder, HIDDEN_DIM).to(device)

train = []
val = []


criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

optimizer = AdamW([
    # {'params': model.encoder.bert.parameters(), 'lr': 2e-6},
    {'params': model.encoder.fc.parameters(), 'lr': 2e-4},
    {'params': model.decoder.parameters(), 'lr': 1e-4}], weight_decay=1e-5,  eps=1e-8)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.1, patience=2, verbose=True
# )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)

def train_model(model, dataloader, optimizer, criterion, epochs=50): # epochs = 50
    best_loss = float('inf')
    patience_counter = 0
    patience = 5
    val_loss = 0

    for epoch in (range(epochs)):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            input_ids, attention_mask, target = batch
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, target, epoch, epochs)
            loss = criterion(outputs.view(-1, VOCAB_SIZE), target.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        val_loss = evaluate_model(model, val_dataloader, criterion)
        scheduler.step(val_loss)
        val.append(val_loss)
        train.append(epoch_loss/len(dataloader))
        print(f'{epoch+1}/{epochs} Train Loss: {epoch_loss/len(dataloader):.4f} valuation Loss:{val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered. Stopping training.")
                break

train_model(model, train_dataloader, optimizer, criterion)

import matplotlib.pyplot as plt
y = np.arange(0, len(val))
plt.plot(y, val)
plt.plot(y, train)
plt.show()

def beam_search_decode_n(model, tokenizer, input_text, beam_width=5, max_text_len = max_len_text, max_len=max_len_summary):
    model.eval()
    with torch.no_grad():
        # Tokenize input text
        tokens_input = tokenizer.encode_plus(
            input_text,
            max_length=max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokens_input["input_ids"].to(device)
        attention_mask = tokens_input["attention_mask"].to(device)

        encoder_outputs = model.encoder(input_ids, attention_mask)
        pooled = encoder_outputs.mean(dim=1)
        hidden = torch.stack([pooled] * model.decoder.lstm.num_layers)
        cell = torch.zeros_like(hidden).to(device)

        start_token_id = tokenizer.bos_token_id
        end_token_id = tokenizer.eos_token_id

        beams = [(0.0, [start_token_id], hidden, cell)]

        for _ in range(max_len):
            new_beams = []
            for score, seq, hidden, cell in beams:
                if seq[-1] == end_token_id:
                    normalized_score = score/len(seq)
                    new_beams.append((normalized_score, seq, hidden, cell))
                    continue

                decoder_input = torch.tensor([seq[-1]], device=device)
                output, hidden_new, cell_new = model.decoder(decoder_input, hidden, cell, encoder_outputs)
                probs = F.log_softmax(output, dim=1)

                top_probs, top_ids = probs.topk(beam_width)

                for i in range(beam_width):
                    new_score = score + top_probs[0][i].item()
                    new_seq = seq + [top_ids[0][i].item()]
                    new_beams.append((new_score, new_seq, hidden_new, cell_new))

            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

        final_seq = beams[0][1]

        summary_tokens = tokenizer.convert_ids_to_tokens(final_seq)
        summary = tokenizer.convert_tokens_to_string(summary_tokens)

        return summary

import random
for _ in range(5):
  i = random.randint(0, x_tr.shape[0])
  summary = beam_search_decode_n(model, tokenizer, x_tr[i], beam_width=5)
  print("text: ", x_tr[i])
  print(f"Summary: {y_tr[i]}")
  print(tokenizer.convert_ids_to_tokens(y_train_seq[i]))
  print("Generated Summary:", summary)
  print()

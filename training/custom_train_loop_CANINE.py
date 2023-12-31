from transformers import CanineTokenizer, CanineForTokenClassification, pipeline, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
import torch
import os

class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_seq_length, label2id):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.label2id = label2id

    def __getitem__(self, index):
        # step 1: tokenize (and adapt corresponding labels)
        sentence = self.data.offer[index]  
        word_labels = self.data.BIO_Tags[index]  
        tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)
        
        # step 2: add special tokens (and corresponding labels)
        tokenized_sentence = ["\ue000"] + tokenized_sentence + ["\ue001"] # add special tokens
        labels.insert(0, "O") # add outside label for \ue000 token
        labels.insert(-1, "O") # add outside label for \ue001 token

        # step 3: truncating/padding
        maxlen = self.max_seq_length

        if (len(tokenized_sentence) > maxlen):
          # truncate
          tokenized_sentence = tokenized_sentence[:maxlen]
          labels = labels[:maxlen]
        else:
          # pad
          tokenized_sentence = tokenized_sentence + ['\x00'for _ in range(maxlen - len(tokenized_sentence))]
          labels = labels + ["O" for _ in range(maxlen - len(labels))]

        # step 4: obtain the attention mask
        attn_mask = [1 if tok != '\x00' else 0 for tok in tokenized_sentence]
        
        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        # encoding = tokenizer(inputs, padding="longest", truncation=True, return_tensors="pt")

        label_ids = [self.label2id[label] for label in labels]
        # the following line is deprecated
        #label_ids = [label if label != 0 else -100 for label in label_ids]
        
        return {
              'ids': torch.tensor(ids, dtype=torch.long),
              'mask': torch.tensor(attn_mask, dtype=torch.long),
              #'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
              'targets': torch.tensor(label_ids, dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []

    sentence = sentence.strip()

    for word, label in zip(sentence.split(), text_labels.split(",")):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

def load_data_split(tokenizer, pth, train_size, label2id, train_batch_size, val_batch_size, max_seq_length):
    df_csv = pd.read_csv(pth, sep=';')
    train_dataset = df_csv.sample(frac=train_size,random_state=200)
    test_dataset = df_csv.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df_csv.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = dataset(train_dataset, tokenizer, max_seq_length, label2id)
    testing_set = dataset(test_dataset, tokenizer, max_seq_length, label2id)

    train_params = {'batch_size': train_batch_size,
                'shuffle': True,
                'num_workers': 0
                }

    test_params = {'batch_size': val_batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    return training_loader, testing_loader

def load_data(tokenizer, pth, label2id, batch_size, max_seq_length):
    df_csv = pd.read_csv(pth, sep=';')
    
    print("FULL Dataset: {}".format(df_csv.shape))

    training_set = dataset(df_csv, tokenizer, max_seq_length, label2id)

    train_params = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 0
                }

    training_loader = DataLoader(training_set, **train_params)
    return training_loader

def load_model(model_pth, device, label2id, id2label):

    model = CanineForTokenClassification.from_pretrained(model_pth, 
                                        num_labels=len(id2label),
                                        id2label=id2label,
                                        label2id=label2id)
    tokenizer = CanineTokenizer.from_pretrained(model_pth)
    return tokenizer, model.to(device)

def valid(model, testing_loader, id2label, device):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['ids'].to(device, dtype = torch.long)
            mask = batch['mask'].to(device, dtype = torch.long)
            targets = batch['targets'].to(device, dtype = torch.long)
            
            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, eval_logits = outputs.loss, outputs.logits
            
            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += targets.size(0)
        
            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")
              
            # compute evaluation accuracy
            flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes \ue000 and \ue001 token predictions)
            active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            eval_labels.extend(targets)
            eval_preds.extend(predictions)
            
            tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy
    
    #print(eval_labels)
    #print(eval_preds)

    labels = [id2label[id.item()] for id in eval_labels]
    predictions = [id2label[id.item()] for id in eval_preds]

    #print(labels)
    #print(predictions)
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return classification_report([labels], [predictions])

def train_loop(model, device, model_save_path, training_loader, testing_loader, id2label, num_epochs, initial_lr, max_grad_norm):
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=initial_lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)  # Using AdamW optimizer

    steps_per_epoch = len(training_loader)
    total_steps = steps_per_epoch * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch in range(num_epochs):
        print(f"Training epoch: {epoch + 1}")
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []
        # put model in training mode
        model.train()
        
        for idx, batch in enumerate(training_loader):
            
            ids = batch['ids'].to(device, dtype = torch.long)
            mask = batch['mask'].to(device, dtype = torch.long)
            targets = batch['targets'].to(device, dtype = torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, tr_logits = outputs.loss, outputs.logits
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            
            if idx % 100==0:
                loss_step = tr_loss/nb_tr_steps
                print(f"Training loss per 100 training steps: {loss_step}")
            
            # compute training accuracy
            flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes \ue000 and \ue001 token predictions)
            active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            tr_preds.extend(predictions)
            tr_labels.extend(targets)
            
            tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy
        
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=max_grad_norm
            )
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}\n")

        report = valid(model, testing_loader, id2label, device)
        print(f"{report}\n")

        model.save_pretrained(os.path.join(model_save_path, f"epoch_{epoch+1}"))
        tokenizer.save_pretrained(os.path.join(model_save_path, f"epoch_{epoch+1}"))

def inference(offer, model, tokenizer):
    pipe = pipeline(task="token-classification", model=model.to("cpu"), tokenizer=tokenizer, aggregation_strategy="simple")
    return pipe(offer)

if __name__ == "__main__":
    model_save_path = '/home/sondors/Documents/price/BERT_NER/weights/CANINE/our_data'
    train_csv_pth = '/home/sondors/Documents/price/BERT_NER/csv/prod_train/train_08our_1opensource.csv'
    test_csv_pth = '/home/sondors/Documents/price/BERT_NER/csv/prod_train/test_02our.csv'
    model_pth = "/home/sondors/29887"
    # model_pth = 'bert-base-uncased'
    device = 'cuda'

    label2id = {'B-width': 1,
                'B-height': 2,
                'B-radius': 3,
                'B-brand': 4,
                'B-line': 5,
                'I-line': 6,
                'O': 0}
    
    id2label = {1: 'B-width',
                2: 'B-height', 
                3: 'B-radius', 
                4: 'B-brand', 
                5: 'B-line', 
                6: 'I-line',
                0: 'O'}
    
    model_pth = "google/canine-c"
    device = 'cuda'

    max_seq_length = 2048
    train_batch_size = 2
    val_batch_size = 1
    initial_lr = 1e-05
    max_grad_norm = 10
    # train_size = 0.9
    num_epochs = 15

    tokenizer, model = load_model(model_pth, device, label2id, id2label)
    training_loader = load_data(tokenizer, train_csv_pth, label2id, train_batch_size, max_seq_length)
    testing_loader = load_data(tokenizer, test_csv_pth, label2id, val_batch_size, max_seq_length)
    train_loop(model, device, model_save_path, training_loader, testing_loader, id2label, num_epochs, initial_lr, max_grad_norm)


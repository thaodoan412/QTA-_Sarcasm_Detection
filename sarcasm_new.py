import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer,TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score


class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


## Test Dataset
class SarcasmTestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings)


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(labels, pred)

    return {"accuracy": accuracy, "f1_score": f1}


def labels(x):
    if x == 0:
        return 0
    else:
        return 1


import re


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)  # no emoji


class CustomTrainer(Trainer):
    def compute_metrics(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.3]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


if __name__ == '__main__':
    # dataset address
    # train = pd.read_csv('https://raw.githubusercontent.com/AmirAbaskohi/SemEval2022-Task6-Sarcasm-Detection/main/Data/Main%20Dataset/Train_Dataset.csv')
    # test = pd.read_csv('https://raw.githubusercontent.com/thaodoan412/QTA-_Sarcasm_Detection/main/cleaned_tweets.csv')
    train = pd.read_csv('./Train_Dataset_minimal.csv')
    test = pd.read_csv('cleaned_tweets_minimal.csv')
    test["sarcastic"] = 1
    test["text"] = test['text'].apply(remove_emoji)
    train_tweets = train['tweet'].values.tolist()
    train_labels = train['sarcastic'].values.tolist()
    test_tweets = test['text'].values.tolist()
    test_labels = test['sarcastic'].values.tolist()
    train_tweets = [str(tweet) for tweet in train['tweet'].values.tolist()]
    test_tweets = [str(tweet) for tweet in test['text'].values.tolist()]
    model_name = 'detecting-Sarcasm'
    task='sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, num_labels=2, loss_function_params={"weight": [0.75, 0.25]})
    train_encodings = tokenizer(train_tweets, truncation=True, padding=True, max_length=512, return_tensors='pt')
    test_encodings = tokenizer(test_tweets, truncation=True, padding=True, max_length=512, return_tensors='pt')

    train_dataset = SarcasmDataset(train_encodings, train_labels)
    test_dataset = SarcasmDataset(test_encodings, test_labels)  # Use SarcasmDataset for test_dataset as well
 
    training_args = TrainingArguments(
        output_dir='./res', evaluation_strategy="steps", num_train_epochs=5, per_device_train_batch_size=32,
        per_device_eval_batch_size=64, warmup_steps=500, weight_decay=0.01, logging_dir='./logs4',
        load_best_model_at_end=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.evaluate()
    preds = trainer.predict(test_dataset=test_dataset)

    probs = torch.from_numpy(preds[0]).softmax(1)

    predictions = probs.numpy()

    newdf = pd.DataFrame(predictions,columns=['Negative_0','Neutral_1','Positive_2'])

    results = np.argmax(predictions,axis=1)

    print(results)

    test['sarcastic'] = 0    
    test_encodings = tokenizer(test_tweets,
                            truncation=True, 
                            padding=True,
                            return_tensors = 'pt')

    f1_score(test_labels, test['sarcastic_result'])
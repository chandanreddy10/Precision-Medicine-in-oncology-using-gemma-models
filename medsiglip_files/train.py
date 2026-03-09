import torch
from torch.optim import AdamW
import pickle
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from dataset import SliceEmbeddingDataset
from torch.utils.data import DataLoader
from model import TwoLayerLinearClassifier
import logging
import numpy as np
from collections import Counter

#Logging 
logging.basicConfig(
    level=logging.INFO,
    filename="train.log",
    format="%(asctime)s | %(levelname)s | %(message)s"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using Device : {device}")

#Load the data and hyperparameters
features_file = "output_embeddings.pkl"
labels_file = "lung_cancer_subtype.pkl"
epochs = 50
batch_size = 512
lr = 0.005

with open(features_file, "rb") as file:
    features = pickle.load(file)

with open(labels_file, "rb") as file:
    labels = pickle.load(file)

logging.info(f"Epochs : {epochs} | Features and Labels Loaded Successfully.")

labels = [label.upper() for label in labels]

map_dict = {"A": "N", "G": "N", "E": "N", "B": "S"}
labels_mapped = [map_dict[label] for label in labels]
unique_classes = sorted(set(labels_mapped))  # ['N', 'S']

class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
inv_class_mapping = {v: k for k, v in class_mapping.items()}

labels_int = [class_mapping[label] for label in labels_mapped]

X_train, X_test, y_train, y_test = train_test_split(
    features, labels_int, test_size=0.10, random_state=42, stratify=labels_int
)

# infer embedding dim from first training sample
sample = X_train[0]  # shape [seq_len, embed_dim]
embed_dim = sample.shape[1]
logging.info(f"Inferred embedding dim: {embed_dim}")

#Dataset and dataloader
train_dataset = SliceEmbeddingDataset(X_train, y_train)
test_dataset = SliceEmbeddingDataset(X_test, y_test)
logging.info(f"Train Sample : {len(train_dataset)} | Test Samples : {len(test_dataset)}")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
num_classes = len(unique_classes)
model = TwoLayerLinearClassifier(
    embed_dim=embed_dim
).to(device)
logging.info("Model Loaded.")

optimizer = AdamW(model.parameters(), lr=lr)

counter = Counter(y_train)  
all_train_labels = []
for series_label, series_emb in zip(y_train, X_train):
    n_slices = series_emb.shape[0]
    all_train_labels.extend([series_label] * n_slices)

counter = Counter(all_train_labels)
num_classes = len(unique_classes)
total = sum(counter.values())

# Weight inversely proportional to frequency
weights = [total / counter[i] for i in range(num_classes)]
weights = torch.tensor(weights, dtype=torch.float32).to(device)
logging.info(f"Class weights: {weights}")
loss_function = nn.CrossEntropyLoss(weight=weights)

#Train and EVal
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    total_batches = 0

    logging.info(f"EPOCH {epoch+1}/{epochs} START")

    for batch_idx, (X, y) in enumerate(train_dataloader):
        X = X.to(device).float()
        y = y.to(device).long()

        optimizer.zero_grad()
        outputs = model(X)

        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

        logging.info(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_dataloader)} | Loss: {loss.item():.6f}")

    avg_train_loss = running_loss / total_batches
    logging.info(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.6f}")

    #Test
    model.eval()
    all_preds = []
    all_targets = []
    test_loss_total = 0.0
    test_batches = 0

    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(device).float()
            y = y.to(device).long()

            outputs = model(X)
            loss = loss_function(outputs, y)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

            test_loss_total += loss.item()
            test_batches += 1

    avg_test_loss = test_loss_total / test_batches
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")

    report = classification_report(
        all_targets,
        all_preds,
        target_names=[inv_class_mapping[i] for i in range(num_classes)],
        digits=4
    )

    logging.info(f"Epoch {epoch+1} Test Loss: {avg_test_loss:.6f}")
    logging.info(f"Epoch {epoch+1} Test Accuracy: {acc:.4f}")
    logging.info(f"Epoch {epoch+1} Test Macro F1: {f1:.4f}")
    logging.info(f"Epoch {epoch+1} Classification Report:\n{report}")

    logging.info(f"EPOCH {epoch+1}/{epochs} END\n")

# Save Model
torch.save(model.state_dict(), "mlp_classifier.pth")
logging.info("Model saved to mlp_classifier.pth")

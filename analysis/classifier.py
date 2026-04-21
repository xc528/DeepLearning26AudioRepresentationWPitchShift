import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report

from embedding_preprocess import EmbeddingPreprocessor
from confusion_matrix import plot_confusion_matrix

INSTRUMENT_CLASSES = {
    '0': 'Clarinet', '1': 'Guitar', '2': 'Female Vocals', '3': 'Flute',
    '4': 'Piano', '5': 'Saxophone', '6': 'Trumpet', '7': 'Violin'
}


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_classes=8, dropout=0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_embeddings(base_dir, model_name, semitones):
    all_embeddings, all_labels = [], []
    for shift in semitones:
        folder = os.path.join(base_dir, model_name, str(shift))
        if not os.path.isdir(folder):
            print(f"Warning: {folder} not found, skipping.")
            continue
        preprocessor = EmbeddingPreprocessor(folder, model_name)
        embeddings, _, instrument_ids = preprocessor.preprocess_embeddings()
        all_embeddings.append(embeddings)
        all_labels.extend(instrument_ids)
    X = torch.cat(all_embeddings, dim=0).float()
    y = torch.tensor([int(i) for i in all_labels], dtype=torch.long)
    return X, y


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (model(X_batch).detach().argmax(1) == y_batch).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * len(y_batch)
        preds = logits.argmax(1)
        correct += (preds == y_batch).sum().item()
        all_preds.append(preds.cpu())
        all_labels.append(y_batch.cpu())
    n = len(loader.dataset)
    return total_loss / n, correct / n, torch.cat(all_preds), torch.cat(all_labels)

def train_classifier(X, y, epochs=100, batch_size=64, lr=1e-3, hidden_dim=512, dropout=0.3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = len(X)
    n_test = int(n * 0.15)
    n_val  = int(n * 0.15)
    n_train = n - n_val - n_test
    train_set, val_set, test_set = random_split(
        TensorDataset(X, y), [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size)
    test_loader  = DataLoader(test_set,  batch_size=batch_size)

    model = MLPClassifier(input_dim=X.shape[1], hidden_dim=hidden_dim,
                          num_classes=8, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()
        _, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(1 - val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | val acc {val_acc:.3f}")

    model.load_state_dict(best_state)
    _, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    print(f"\nTest accuracy: {test_acc:.4f}")
    return model, y_true, y_pred


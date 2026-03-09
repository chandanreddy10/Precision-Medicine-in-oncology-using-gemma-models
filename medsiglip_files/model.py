import torch
import torch.nn as nn

class TwoLayerLinearClassifier(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim=128,
        num_classes=2,
        dropout=0.2
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        x: [batch_size, seq_len, embed_dim]
        Mean pooling over sequence dimension.
        """
        logits = self.classifier(x)
        return logits

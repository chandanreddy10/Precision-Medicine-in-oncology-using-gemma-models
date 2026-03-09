import torch
from torch.utils.data import Dataset

class SliceEmbeddingDataset(Dataset):
    """
    Treat each slice embedding as a separate sample.
    Input: list of series embeddings [num_slices, embed_dim]
    Output: each slice [embed_dim] with the series label
    """
    def __init__(self, embeddings_list, labels_list):
        """
        embeddings_list: list of np.arrays or tensors, each shape [num_slices, embed_dim]
        labels_list: list of integers, one per series
        """
        self.data = []
        self.labels = []

        for series_emb, label in zip(embeddings_list, labels_list):
            if not isinstance(series_emb, torch.Tensor):
                series_emb = torch.tensor(series_emb, dtype=torch.float32)
            # Append each slice as a separate sample
            for slice_emb in series_emb:
                self.data.append(slice_emb)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]   # shape [embed_dim]
        y = self.labels[idx] # scalar
        return x, y
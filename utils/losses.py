import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, emb1, emb2, labels):
        """
        emb1, emb2: embeddings from the two branches of Siamese network
        labels: 1 if same class, 0 if different class
        """
        # Compute Euclidean distance
        dist = F.pairwise_distance(emb1, emb2)
        
        # Contrastive loss
        loss = torch.mean(labels * torch.pow(dist, 2) + 
                         (1 - labels) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        
        return loss

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(NTXentLoss, self).__init__()
        self.temperature = max(temperature, 0.01)  # prevent division by zero
    
    def forward(self, z_i, z_j):
        """
        NT-Xent loss for self-supervised learning
        z_i, z_j: embeddings from positive pairs
        Uses a simpler and more stable formulation
        """
        batch_size = z_i.shape[0]
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1, p=2)
        z_j = F.normalize(z_j, dim=1, p=2)
        
        # Simple contrastive loss: maximize similarity between z_i and z_j
        # Use cosine similarity (dot product of normalized vectors)
        sim_pos = torch.sum(z_i * z_j, dim=1) / self.temperature  # (B,)
        
        # Compute negative similarities
        z_all = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        
        # Compute full similarity matrix
        sim_matrix = torch.mm(z_all, z_all.t()) / self.temperature  # (2B, 2B)
        sim_matrix = torch.clamp(sim_matrix, min=-10, max=10)
        
        # Create labels: positive pairs at (0, B) to (B-1, 2B-1) and (B, 0) to (2B-1, B-1)
        # Mask out self-similarities and the diagonal
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_all.device)
        
        # For each i, the positive is at i+B (mod 2B)
        # Compute log softmax over all samples
        log_softmax = torch.nn.functional.log_softmax(sim_matrix, dim=1)
        
        # Positive pair indices: (0,B), (1,B+1), ..., (B-1,2B-1), (B,0), ..., (2B-1, B-1)
        pos_idx_col = torch.cat([torch.arange(batch_size, 2*batch_size, device=z_all.device),
                                  torch.arange(batch_size, device=z_all.device)])
        pos_idx_row = torch.arange(2*batch_size, device=z_all.device)
        
        loss = -log_softmax[pos_idx_row, pos_idx_col].mean()
        
        return loss

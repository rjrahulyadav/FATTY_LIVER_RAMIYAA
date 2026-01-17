import torch
import torch.nn as nn
from torchvision.models import resnet50

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        # Load pre-trained ResNet-50 and modify it
        try:
            self.encoder = resnet50(weights='DEFAULT')  # Updated for newer torchvision
        except:
            # Fallback for older torchvision versions
            self.encoder = resnet50(pretrained=True)
        
        # Remove the fully connected layer
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Classification head for downstream task with regularization
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)  # 5 classes
        )
        
        self.embedding_dim = embedding_dim
        
    def forward_once(self, x):
        # Forward pass through encoder
        features = self.encoder(x)
        features = features.view(features.size(0), -1)  # Flatten
        return features
    
    def forward(self, x1, x2=None):
        # If two inputs, return embeddings for contrastive loss
        if x2 is not None:
            emb1 = self.forward_once(x1)
            emb2 = self.forward_once(x2)
            proj1 = self.projection_head(emb1)
            proj2 = self.projection_head(emb2)
            return proj1, proj2
        # If single input, return logits for classification
        else:
            features = self.forward_once(x1)
            logits = self.classifier(features)
            return logits

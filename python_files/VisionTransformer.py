import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    """
    A VisionTransformer as described in "An image is worth 16x16 words"
    """

    def __init__(self, embedding, d, num_layers, num_heads, ff_dim, num_classes, dropout_p=0.1):
        """
        :param embedding: PatchEmbedding instance
        :param d: Embedding dimension
        :param num_layers: Number of transformer encoder layers
        :param num_heads: Number of heads in each transformer block
        :param ff_dim: Hidden dimension of feedforward layer in each trasformer block
        :param num_classes: Number of classification classes
        :param dropout_p: Dropout probability
        """
        if d != embedding.d:
            raise Exception("Embedding dimension do not fit")

        super().__init__()
        self.d = d
        self.patch_embedding = embedding
        self.class_token = nn.Parameter(torch.zeros(1, 1, d))
        self.position_embedding = nn.Parameter(torch.zeros(1, embedding.N + 1, d))
        tranformer_enc_layer = nn.TransformerEncoderLayer(d, num_heads, ff_dim, dropout_p,
                                                          activation="gelu", batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(tranformer_enc_layer, num_layers)
        self.position_dropout = nn.Dropout(p=dropout_p)
        self.norm = nn.LayerNorm(d, 1e-6)
        self.classification_head = nn.Linear(d, num_classes)
        self.sm = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass
        :param x: Input images. Shape: (batch_size, img_c, img_h, img_w)
        :return: Output classification probabilities. Shape: (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        x = self.patch_embedding(x)
        x = torch.concat((self.class_token.expand(batch_size, -1, -1), x), dim=1)
        x = x + self.position_embedding
        x = self.position_dropout(x)
        x = self.transformer_enc(x)
        x = self.classification_head(x[:, 0])
        return x

    def setClassificationHead(self, num_classes):
        """
        Sets a new classification head for finetuning
        :param num_classes: The number of classes for classification
        :return: None
        """
        self.classification_head = nn.Linear(self.d, num_classes)

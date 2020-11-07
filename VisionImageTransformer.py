import torch 
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dims,
        heads,
        dropout
    ):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.embedding_dims = embedding_dims
        self.head_dims = int(embedding_dims/heads)

        self.key = nn.Linear(self.head_dims, self.head_dims)
        self.query = nn.Linear(self.head_dims, self.head_dims)
        self.value = nn.Linear(self.head_dims, self.head_dims)

        self.fc = nn.Linear(self.head_dims*self.heads, self.embedding_dims)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask):
        Batch = query.shape[0]

        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]

        query = query.reshape(Batch, query_len, self.heads, self.head_dims)
        key = key.reshape(Batch, key_len, self.heads, self.head_dims)
        value = value.reshape(Batch, value_len, self.heads, self.head_dims)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        attention_score = torch.einsum('bqhd,bkhd->bhqk', [query, key])

        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, float('-1e20'))

        attention_score = attention_score/((self.head_dims)**(1/2))
        attention_score = torch.softmax(attention_score, dim=-1)

        out = torch.einsum('bhqv,bvhd->bqhd', [attention_score, value]).reshape(
            Batch, query_len, self.heads*self.head_dims
        )

        out = self.dropout(self.fc(out))

        return out



class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dims,
        heads,
        dropout, 
        forward_expansion,
        layer_norm_eps
    ):
        super(TransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.attention = SelfAttention(embedding_dims, heads, dropout)
        self.feed_forward = nn.Sequential(
                nn.Linear(embedding_dims, embedding_dims*forward_expansion),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dims*forward_expansion, embedding_dims),
                nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        norm = self.layer_norm1(x)
        attention_block = self.attention(norm, norm, norm, mask)
        add = x + attention_block
        norm = self.layer_norm2(add)
        feed_forward = self.feed_forward(norm)
        out = feed_forward + add
        return out


class ViT(nn.Module):
    def __init__(
        self,
        patch_height,
        patch_width,
        max_len,
        embedding_dims,
        heads,
        forward_expansion,
        num_layers,
        dropout,
        layer_norm_eps,
        num_classes
    ):
        super(ViT, self).__init__()
        
        self.vit_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dims,
                    heads,
                    dropout,
                    forward_expansion,
                    layer_norm_eps
                )
                for _ in range(num_layers)
            ]
            
        )
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, embedding_dims))
        self.patch_embeddings = nn.Linear(embedding_dims, embedding_dims)
        self.postional_embedding = nn.Parameter(torch.zeros(1, max_len+1, embedding_dims))
        self.to_cls_token = nn.Identity()
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dims),
            nn.Linear(embedding_dims, num_classes*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_classes*4, num_classes)
        )
        self.dropout = nn.Dropout(dropout)


    def forward(self, images):
        patches = images.unfold(2, self.patch_height, self.patch_width).unfold(3, self.patch_height, self.patch_width)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(
            patches.shape[0],
            patches.shape[1],
            patches.shape[2],
            patches.shape[3]*patches.shape[4]*patches.shape[5]
        )
        patches = patches.view(patches.shape[0], -1, patches.shape[-1])

        x = self.cls_embedding.expand(patches.shape[0], -1, -1)
        patch_embeddings = self.patch_embeddings(patches)
        x = torch.cat((x, patch_embeddings), dim=1) + self.postional_embedding
        x = self.dropout(x)
        mask = None
        for block in self.vit_blocks:
            x = block(x, mask)
        out = self.to_cls_token(x[:, 0])
        out = self.classifier(out)
        return out



if __name__ == "__main__":

    model = ViT(
        patch_height = 16,
        patch_width = 16,
        embedding_dims = 768,
        dropout = 0.1,
        heads = 4,
        num_layers = 4,
        forward_expansion = 4,
        max_len = int((32*32)/(16*16)),
        layer_norm_eps = 1e-5,
        num_classes = 10,
    )

    a = torch.randn(32, 3, 32, 32)
    output = model(a)
    print(output.shape)
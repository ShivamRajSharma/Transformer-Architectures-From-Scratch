from time import time
import torch 
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dims,
        heads
    ):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.embedding_dims = embedding_dims
        self.head_dims = int(embedding_dims/heads)

        self.key = nn.Linear(self.head_dims, self.head_dims)
        self.query = nn.Linear(self.head_dims, self.head_dims)
        self.value = nn.Linear(self.head_dims, self.head_dims)

        self.fc = nn.Linear(self.head_dims*self.heads, self.embedding_dims)
    
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

        attention_score = attention_score//((self.head_dims)**(1/2))
        attention_score = torch.softmax(attention_score, dim=-1)

        out = torch.einsum('bhqv,bvhd->bqhd', [attention_score, value]).reshape(
            Batch, query_len, self.heads*self.head_dims
        )

        out = self.fc(out)

        return out



class BertBlock(nn.Module):
    def __init__(
        self,
        embedding_dims,
        heads,
        dropout, 
        forward_expansion,
        layer_norm_eps
    ):
        super(BertBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.attention = SelfAttention(embedding_dims, heads)
        self.feed_forward = nn.Sequential(
                nn.Linear(embedding_dims, embedding_dims*forward_expansion),
                nn.GELU(),
                nn.Linear(embedding_dims*forward_expansion, embedding_dims)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        attention_block = self.attention(x, x, x, mask)
        add = self.dropout(self.layer_norm1(x + attention_block))
        feed_forward = self.feed_forward(add)
        out = self.dropout(self.layer_norm2(feed_forward + add))
        return out



class Embeddings(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        embedding_dims
    ):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dims)
        self.positional_embeddings = nn.Parameter(
            torch.zeros(1, max_len, embedding_dims)
        )
        self.segment_embeddings = nn.Embedding(3, embedding_dims)
    
    def forward(self, x, segment_x):
        sentence_len = x.shape[1]
        word_embeddings = self.word_embeddings(x)
        positional_embeddings = self.positional_embeddings[:, :sentence_len, :]
        segment_embeddings = self.segment_embeddings(segment_x)
        return word_embeddings + positional_embeddings + segment_embeddings



class BERT(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        mask_idx,
        embedding_dims,
        heads,
        forward_expansion,
        num_layers,
        dropout,
        layer_norm_eps
    ):
        super(BERT, self).__init__()
        self.embedding = Embeddings(
            vocab_size,
            max_len,
            embedding_dims
        )
        
        self.bert_blocks = nn.Sequential(
            *[
                BertBlock(
                    embedding_dims,
                    heads,
                    dropout,
                    forward_expansion,
                    layer_norm_eps
                )
                for _ in range(num_layers)
            ]
            
        )

        self.layer_norm = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.fc = nn.Linear(embedding_dims, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.mask_idx = mask_idx

        self.apply(self._init_weight)
    
    # @hugging_face
    def _init_weight(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
        
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def create_mask(self, mask):
        mask  = (mask != self.mask_idx).unsqueeze(1).unsqueeze(2)
        return mask
        
    def forward(self, x, segment_x, mask):
        mask = self.create_mask(mask)
        x = self.dropout(self.embedding(x, segment_x))
        for block in self.bert_blocks:
            x = block(x, mask)
        return x


if __name__ == '__main__':
    #DEFAULT BERT PARAMETER :-
    vocab_size = 30522
    embedding_dims = 768
    dropout = 0.1
    heads = 12
    num_layers = 12
    forward_expansion = 4
    max_len = 512
    layer_norm_eps = 1e-12
    mask_idx = 0

    x = torch.randint(1, 100, (32, 100))
    x_segment = torch.randint(0, 2, (32, 100))

    model = BERT(
        vocab_size,
        max_len,
        mask_idx,
        embedding_dims,
        heads,
        forward_expansion,
        num_layers,
        dropout,
        layer_norm_eps
    )

    mask = torch.randint(0, 2, (32, 100))
    start = time()
    y = model(x, x_segment, mask)
    print(f'INFERENCE TIME = {time() - start}sec')
    x = sum(p.numel() for p in model.parameters())
    print(f'NUMBER OF PARAMETERS ARE = {x}')

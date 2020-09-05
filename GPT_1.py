from time import time
import torch 
import torch.nn as nn 

class SelfAttention(nn.Module):
    def __init__(self, input_dims, heads):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dims = int(input_dims/heads)
        self.input_dims = input_dims

        self.query = nn.Linear(self.head_dims, self.head_dims)
        self.key = nn.Linear(self.head_dims, self.head_dims)
        self.value = nn.Linear(self.head_dims, self.head_dims)
        self.fc = nn.Linear(self.head_dims*heads, self.input_dims)
    
    def forward(self, query, key, value, mask):
        Batch, Seq_len, embed = query.shape
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]

        query = query.reshape(Batch, query_len, self.heads, self.head_dims)
        key = key.reshape(Batch, key_len, self.heads, self.head_dims)
        value = value.reshape(Batch, value_len, self.heads, self.head_dims)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        score = torch.einsum('bqhd,bkhd->bhqk', [query, key])
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-1e20'))
        
        attention_score = nn.Softmax(dim=-1)(score/((self.head_dims)**(1/2)))
        out = torch.einsum('bhqv,bvhd->bqhd', [attention_score, value]).reshape(Batch, query_len, self.head_dims*self.heads)
        out = self.fc(out)
        
        return out


class GPTBlock(nn.Module):
    def __init__(
        self,
        heads,
        embedding_dims,
        dropout,
        forward_expansion, 
        layer_norm_eps
    ):
        super(GPTBlock, self).__init__()
        self.embedding_dims = embedding_dims
        self.attention = SelfAttention(embedding_dims, heads)
        self.layer_norm1 = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.feed_forward = nn.Sequential(
            *[
                nn.Linear(embedding_dims, embedding_dims*forward_expansion),
                nn.GELU(),
                nn.Linear(embedding_dims*forward_expansion, embedding_dims)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_block = self.attention(x, x, x, mask)
        add = self.dropout(self.layer_norm1(attention_block + x))
        feed_forward = self.feed_forward(add)
        out = self.dropout(self.layer_norm2(feed_forward + add))
        return out


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dims,
        dropout,
        heads,
        num_of_layers,
        forward_expansion,
        max_len,
        layer_norm_eps = 1e-5
    ):
        super(GPT, self).__init__()
        self.embedding_dims = embedding_dims
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dims)
        self.positional_embeddings = nn.Parameter(torch.zeros(1, max_len, embedding_dims))
        self.dropout = nn.Dropout(dropout)
        self.gpt_blocks = nn.ModuleList(
            [
                GPTBlock(
                    heads,
                    embedding_dims,
                    dropout,
                    forward_expansion,
                    layer_norm_eps

                )
                for _ in range(num_of_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.fc = nn.Linear(embedding_dims, vocab_size)

        self.apply(self._init_weights)
    
    #From @HuggingFace
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def casual_mask(self, x):
        mask = torch.tril(torch.ones((x.shape[0], x.shape[-1], x.shape[-1]))).unsqueeze(1)
        return mask

    def forward(self, x):
        casual_mask = self.casual_mask(x)
        seq_len = x.shape[-1]
        word_embeddings = self.word_embeddings(x)
        x = self.dropout(word_embeddings + self.positional_embeddings[:, :seq_len, :])
        for block in self.gpt_blocks:
            x = block(x, casual_mask)
        x = self.layer_norm(x)
        out = self.fc(x)
        return x
        

if __name__ == '__main__':
    #DEFAULT GPT PARAMETERS :-
    vocab_size = 40478
    embedding_dims = 768
    dropout = 0.1
    heads = 12
    num_of_layers = 12
    forward_expansion = 4
    max_len = 512


    a = torch.randint(1, 100, (1, 300))
    model = GPT(
        vocab_size,
        embedding_dims,
        dropout,
        heads,
        num_of_layers,
        forward_expansion,
        max_len,
    )

    start = time()
    y = model(a)
    print(f'INFERENCE TIME = {time() - start}sec')
    x = sum(p.numel() for p in model.parameters())
    print(f'NUMBER OF PARAMETERS ARE = {x}')

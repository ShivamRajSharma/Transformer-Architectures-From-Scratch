from time import time
import torch 
import torch.nn as nn 

class SelfAttention(nn.Module):
    def __init__(self, input_dims, heads):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dims = int(input_dims/heads)
        self.input_dims = input_dims

        self.expand = nn.Linear(self.input_dims, self.input_dims*3)
        self.fc = nn.Linear(self.head_dims*heads, self.input_dims)
    
    def split_(self, x):
        query, key, value = x.split(self.input_dims, dim=-1)
        return query, key, value

    def forward(self, x, mask, past):
        Batch, seq_len, embed = x.shape
        expand = self.expand(x)
        query, key, value = self.split_(expand)
        
        query = query.reshape(Batch, seq_len, self.heads, self.head_dims)
        key = key.reshape(Batch, seq_len, self.heads, self.head_dims)
        value = value.reshape(Batch, seq_len, self.heads, self.head_dims)

        present = torch.cat((key.unsqueeze(0), value.unsqueeze(0)), dim=0)

        if past is not None:
            past_key, past_value = past
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
        
        score = torch.einsum('bqhd,bkhd->bhqk', [query, key])
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-1e20'))
        
        attention_score = nn.Softmax(dim=-1)(score/((self.head_dims)**(1/2)))
        out = torch.einsum('bhqv,bvhd->bqhd', [attention_score, value]).reshape(Batch, seq_len, self.head_dims*self.heads)
        out = self.fc(out)
        
        return out, present


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

    def forward(self, x, mask, past):
        attention_block, present = self.attention(self.layer_norm1(x), mask, past)
        add = self.dropout(self.layer_norm2(attention_block + x))
        feed_forward = self.feed_forward(add)
        out = self.dropout(feed_forward + add)
        return out, present


class GPT2(nn.Module):
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
        super(GPT2, self).__init__()
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

    
    def casual_mask(self, x, past):
        ones_matix = torch.ones((x.shape[-1], x.shape[-1]))
        mask = torch.tril(ones_matix)
        if past is not None:
            mask = torch.cat((ones_matix, mask), dim=1)
        mask = mask.unsqueeze(0).unsqueeze(1)
        return mask
    

    def forward(self, x, past=None):
        casual_mask = self.casual_mask(x, past)
        seq_len = x.shape[-1]
        word_embeddings = self.word_embeddings(x)
        x = self.dropout(word_embeddings + self.positional_embeddings[:, :seq_len, :])
        presents = []
        past_layer = None
        for num, block in enumerate(self.gpt_blocks):
            if past is not None:
                past_layer = past[num]
            x, present = block(x, casual_mask, past_layer)
            presents.append(present)
        return x, presents
        

if __name__ == '__main__':
    #DEFAULT GPT-2 PARAMETERS :-
    vocab_size = 50257
    embedding_dims = 768
    dropout = 0.1
    heads = 12
    num_of_layers = 12
    forward_expansion = 4
    max_len = 1024


    a = torch.randint(1, 100, (3, 300))
    model = GPT2(
        vocab_size,
        embedding_dims,
        dropout,
        heads,
        num_of_layers,
        forward_expansion,
        max_len,
    )

    start = time()
    past_key_value = None
    for i in range(2):
        y, past_key_value = model(a, past_key_value)
    print(f'INFERENCE TIME = {time() - start}sec')
    x = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'NUMBER OF PARAMETERS ARE = {x}')

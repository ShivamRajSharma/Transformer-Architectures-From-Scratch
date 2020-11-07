from time import time
import torch 
import torch.nn as nn 

class Attention(nn.Module):
    def __init__(self, input_shape, head):
        super(Attention, self).__init__()
        self.head = head
        self.input_shape = input_shape
        self.head_dims = int(input_shape // head)

        self.query = nn.Linear(self.head_dims, self.head_dims)
        self.key = nn.Linear(self.head_dims, self.head_dims)
        self.value = nn.Linear(self.head_dims, self.head_dims)
        self.fc = nn.Linear(self.head_dims*head, input_shape)

    def forward(self, query, key, value, mask=None):
        batch = query.shape[0]
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]
        
        query = query.reshape(batch, query_len, self.head, self.head_dims)
        key = key.reshape(batch, key_len, self.head, self.head_dims)
        value = value.reshape(batch, value_len, self.head, self.head_dims)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        score = torch.einsum("bqhd,bkhd->bhqk", [query, key])
        
        if mask is not None:
            score.masked_fill(mask == 0, float("-1e20"))
        score = torch.softmax(score/((self.head_dims)**(1/2)), dim=-1)
        
        out = torch.einsum("bhqv,bvhd->bqhd", [score, value])
        out = out.reshape(batch, query_len, self.head*self.head_dims)
        out = self.fc(out)
        
        return out



class TransformerBlock(nn.Module):
    def __init__(self, input_shape, head, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = Attention(input_shape, head)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_shape, input_shape*forward_expansion),
            nn.GELU(),
            nn.Linear(input_shape*forward_expansion, input_shape)
        )
        self.layernorm1 = nn.LayerNorm(input_shape)
        self.layernorm2 = nn.LayerNorm(input_shape)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask):
        attention = self.attention(query, key, value, mask)
        add = attention + query 
        regulazation = self.dropout(self.layernorm1(add))
        forward = self.feed_forward(regulazation)
        out = self.dropout(self.layernorm2(forward + regulazation))
        return out



class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_out,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_len
    ):
        super(Encoder, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_out)
        self.postional_embedding =  nn.Parameter(torch.zeros(1, max_len, embedding_out))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_out,
                    heads,
                    dropout,
                    forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        
    def forward(self, x, mask):
        word_embedding = self.word_embedding(x)
        postional_embedding = self.postional_embedding[:, :x.shape[1], :]
        out = self.dropout(word_embedding + postional_embedding)
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out



class DecoderBlock(nn.Module):
    def __init__(
        self,
        embedding_out,
        head,
        forward_expansion,
        dropout
    ):
        super(DecoderBlock, self).__init__()
        self.attention = Attention(embedding_out, head)
        self.transformer_block = TransformerBlock(
            embedding_out, 
            head, 
            dropout, 
            forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_out)

    def forward(self, query, key, value, src_mask, causal_mask):
        attention = self.attention(query, query, query, causal_mask)
        query = self.dropout(self.norm(attention + query))
        out = self.transformer_block(query, key, value, src_mask)
        return out



class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_out,
        num_layers,
        head,
        forward_expansion,
        dropout,
        max_len
    ):
        super(Decoder, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_out)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_len, embedding_out))
        self.layers = nn.Sequential(
            *[
            DecoderBlock(
                embedding_out,
                head,
                forward_expansion,
                dropout
            )
            for _ in range(num_layers)
        ]
        )
        self.fc = nn.Linear(embedding_out, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask, casual_mask):
        x = self.dropout(self.word_embedding(x) + self.positional_embedding[:, :x.shape[1], :])
        for layer in self.layers:
            x = layer(
                x, 
                encoder_output, 
                encoder_output, 
                src_mask, 
                casual_mask
            )
        out = self.fc(x)
        return out



class Transformers(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        pad_idx,
        embedding_out,
        num_layers,
        forward_expansion,
        head,
        dropout,
        max_len
    ):
        super(Transformers, self).__init__()
        self.encoder = Encoder(
            input_vocab_size,
            embedding_out,
            num_layers,
            head,
            forward_expansion,
            dropout,
            max_len
        )
        
        self.decoder = Decoder(
            output_vocab_size,
            embedding_out,
            num_layers,
            head,
            forward_expansion,
            dropout,
            max_len
        )
        
        self.pad_idx = pad_idx
        self.apply(self._init_weights)

    #From @HuggingFace
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def pad_mask(self, inputs):
        pad_mask = (inputs != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return pad_mask

    def causal_mask(self, target):
        N, target_len = target.shape
        target_mask = torch.tril(torch.ones((N, target_len, target_len))).unsqueeze(1)
        return target_mask

    def forward(self, inputs, target):
        pad_mask = self.pad_mask(inputs)
        causal_mask = self.causal_mask(target)
        encoder_output = self.encoder(inputs, pad_mask)
        decoder_out = self.decoder(target, encoder_output, pad_mask, causal_mask)
        return decoder_out
        


if __name__ == "__main__":
    #Depends on the Tokenizer
    input_vocab_size = 100
    output_vocab_size = 200

    #DEFAULT TRANSFORMERS PARAMETERS:-
    pad_idx = 0 
    embedding_out = 512
    num_layers = 6
    forward_expansion = 4
    head = 8
    dropout = 0.1
    max_len = 512

    inputs = torch.randint(0, 100, (32, 200))
    targets = torch.randint(0, 100, (32,100))

    model = Transformers(
        input_vocab_size,
        output_vocab_size,
        pad_idx,
        embedding_out,
        num_layers,
        forward_expansion,
        head,
        dropout,
        max_len
    )

    start = time()
    y = model(inputs, targets)
    print(f'INFERENCE TIME = {time() - start}sec')
    x = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'NUMBER OF PARAMETERS ARE = {x}')

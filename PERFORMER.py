from time import time
import torch 
import torch.nn as nn 


class FastAttention(nn.Module):
    def __init__(self, input_shape, head, n_features):
        super(FastAttention, self).__init__()
        self.head = head
        self.input_shape = input_shape
        self.depth = int(input_shape // head)
        self.n_features = n_features
        self.key_ORF = self.OrthogonalRandomFeature()
        self.query_ORF = self.OrthogonalRandomFeature()

        self.query = nn.Linear(self.depth, self.depth)
        self.key = nn.Linear(self.depth, self.depth)
        self.value = nn.Linear(self.depth, self.depth)
        self.fc = nn.Linear(self.depth*head, input_shape)

    def kernel_function(self, x, flag):
        ORF = self.query_ORF if flag == 'query' else self.key_ORF
        normalization_factor = 1/ORF.shape[-1]**0.25
        x *= normalization_factor
        out = torch.einsum('nhsd, fd -> nhsf', x, ORF)
        kernel_fn = nn.ReLU()(out) + 1e-3
        return kernel_fn

    def OrthogonalRandomFeature(self):
        n = self.n_features//self.depth
        remainder = self.n_features%self.depth
        orthogonal_features = []
        for _ in range(n):
            normal_feature = torch.rand(self.depth, self.depth)
            orthogonal_feature, _ = torch.qr(normal_feature)
            orthogonal_features.append(orthogonal_feature)
        
        if remainder > 0 :
            normal_feature = torch.rand(self.depth, self.depth)
            orthogonal_feature, _ = torch.qr(normal_feature)
            orthogonal_features.append(orthogonal_feature[0: remainder])
        
        orthogonal_features = torch.cat(orthogonal_features)
        mutilplier =  torch.randn(self.n_features, self.depth).norm(dim=1)
        final_features = torch.matmul(torch.diag(mutilplier), orthogonal_features)

        return final_features

    def causal_attention(self, q, k, v):
        k_cumsum = k.cumsum(dim=-2)
        x = torch.einsum('nhkf, nhkd -> nhkfd', k, v)
        x = x.cumsum(dim=-3)
        x /= k_cumsum.unsqueeze(dim=-1)
        out = torch.einsum('nhqfd, nhqf -> nhqd', x, q)
        return out

    
    def bidirectional_attention(self, q, k, v):
        kt_i = torch.einsum('nhkf -> nhf', k)
        normalization_factor = 1/(torch.einsum('nhqf, nhf -> nhq', q, kt_i))
        k_v = torch.einsum('nhkf, nhkd -> nhfd', k, v)
        attention = torch.einsum('nhfd, nhqf,  nhq-> nhqd', k_v, q, normalization_factor)
        return attention


    def forward(self, query, key, value, mask=None, casual_mask=False):
        batch = query.shape[0]
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]

        
        query = query.reshape(batch, query_len, self.head, self.depth)
        key = key.reshape(batch, key_len, self.head, self.depth)
        value = value.reshape(batch, value_len, self.head, self.depth)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        
        if mask is not None:
            key.masked_fill(mask == 0, float("-1e20"))
        
        query = self.kernel_function(query, 'query')
        key  = self.kernel_function(key, 'key')
        
        if casual_mask:
            out = self.causal_attention(query, key, value)
        else:
            out = self.bidirectional_attention(query, key, value)
            
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(batch, query_len, self.head*self.depth)
        out = self.fc(out)
        
        return out



class PerformerBlock(nn.Module):
    def __init__(self, input_shape, head, n_features, dropout, forward_expansion):
        super(PerformerBlock, self).__init__()
        self.attention = FastAttention(input_shape, head, n_features)
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
        n_features,
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
                PerformerBlock(
                    embedding_out,
                    heads,
                    n_features,
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
        n_features,
        forward_expansion,
        dropout
    ):
        super(DecoderBlock, self).__init__()
        self.attention = FastAttention(embedding_out, head, n_features)
        self.Performer_block = PerformerBlock(
            embedding_out, 
            head, 
            n_features,
            dropout, 
            forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_out)

    def forward(self, query, key, value, src_mask):
        attention = self.attention(query, query, query, src_mask, True)
        query = self.dropout(self.norm(attention + query))
        out = self.Performer_block(query, key, value, src_mask)
        return out



class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_out,
        num_layers,
        head,
        n_features,
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
                n_features,
                forward_expansion,
                dropout
            )
            for _ in range(num_layers)
        ]
        )
        self.fc = nn.Linear(embedding_out, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask):
        x = self.dropout(self.word_embedding(x) + self.positional_embedding[:, :x.shape[1], :])
        for layer in self.layers:
            x = layer(
                x, 
                encoder_output, 
                encoder_output, 
                src_mask
            )
        out = self.fc(x)
        return out



class Performers(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        pad_idx,
        embedding_out,
        num_layers,
        forward_expansion,
        head,
        n_features,
        dropout,
        max_len
    ):
        super(Performers, self).__init__()
        self.encoder = Encoder(
            input_vocab_size,
            embedding_out,
            num_layers,
            head,
            n_features,
            forward_expansion,
            dropout,
            max_len
        )
        
        self.decoder = Decoder(
            output_vocab_size,
            embedding_out,
            num_layers,
            head,
            n_features,
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
    
    def input_pad_mask(self, inputs):
        pad_mask = (inputs != self.pad_idx).unsqueeze(1).unsqueeze(3)
        return pad_mask
    
    def  output_pad_mask(self, targets):
        pad_mask = (targets != self.pad_idx).unsqueeze(1).unsqueeze(3)

    def forward(self, inputs, target):
        input_pad_mask = self.input_pad_mask(inputs)
        output_pad_mask = self.output_pad_mask(targets)
        encoder_output = self.encoder(inputs, input_pad_mask)
        # return encoder_output
        decoder_out = self.decoder(target, encoder_output, output_pad_mask)
        return decoder_out
        


if __name__ == "__main__":
    #Depends on the Tokenizer
    input_vocab_size = 100
    output_vocab_size = 200

    #DEFAULT PerFORMERS PARAMETERS:-
    pad_idx = 0 
    embedding_out = 512
    num_layers = 6
    forward_expansion = 4
    head = 8
    n_features = 256
    dropout = 0.1
    max_len = 512

    inputs = torch.randint(0, 100, (32, 200))
    targets = torch.randint(0, 100, (32,100))

    model = Performers(
        input_vocab_size,
        output_vocab_size,
        pad_idx,
        embedding_out,
        num_layers,
        forward_expansion,
        head,
        n_features,
        dropout,
        max_len
    )

    start = time()
    y = model(inputs, targets)
    print(f'INFERENCE TIME = {time() - start}sec')
    x = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'NUMBER OF PARAMETERS ARE = {x}')
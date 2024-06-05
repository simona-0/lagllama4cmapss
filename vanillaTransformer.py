import torch
import torch.nn as nn

class SelfAttention(nn.Module): # building blcok attention mechanism
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads # multi-head attention (normally 8) run parallely
        self.head_dim = embed_size//heads

        assert(self.head_dim * heads == embed_size)
       
        # in multi-head attention, the query (Q), key (K), and value (V) matrices 
        # are SPLIT into multiple 'heads' or parallel attention mechanisms. 
        # each head performs the attention computation independently on the same input, 
        # allowing the model to attend to different representation subspaces of the input in parallel.

        # all plurals: values, keys, queries

        # passing these tensors through linear transformation layers 
        # to project them into different subspaces
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
        # Attention(Q, K, V) = softmax(Q*K^T/sqrt(d_k))*V

    def forward(self, values, keys, query, mask):
    # In PyTorch, the forward method of a module (like SelfAttention) is automatically called 
    # when you treat the module as a function and pass input tensors to it. 

        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # splitting embedding into self.heads pieces, why? more heads more efficiency
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # energy = Q*K^T
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len) head_dim (d) eliminated

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20")) 
        # replaces the elements in energy with negative infinity
        # at the positions where the corresponding elements in mask == 0 are True

        # the mask is useful in seq2seq model because it prevents the model from 
        # attending to future tokens when predicting the next token

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3) 
        # dim=3? 
        # apply softmax along the 4th (zero-based indexing) dimension of energy

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape( # *V
            N, query_len, self.head * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len) same as energy shape
        # values shape: ((N, value_len, heads, head_dim)
        # out shape: (N, query_len, heads, head_dim) 
        # (value_len == key_len == l) why tho ???
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module): # Encoder
    def __init__(self, embed_size, heads, dropout, forward_expansion): 
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size) # normalization layer in the Add&Norm layer
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size), # forward_expansion???
            # to increase the dimensionality of the input 
            # before applying the non-linear activation function
            nn.ReLU(), 
            nn.Linear(forward_expansion*embed_size, embed_size)
        ) 
        # the larger the forward_expansion value, the more parameters the FFN layer will have, 
        # and the more expressive power the model can potentially gain.
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask): 
        attention = self.attention(value, key, query, mask) 
        # outcome after the multi-head attention block

        x = self.dropout(self.norm1(attention + query)) # query as value, query as key, query as query
        forward = self.feed_forward(x)
        # outcome after the feed forward block
        out = self.dropout(self.norm2(forward + x))
        return out
                           
class Encoder(nn.Module): # Word Embedding + Positional Encoding + TransformerBlock x N
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers, # x N
            heads,
            device, # cuda
            forward_expansion,
            dropout,
            max_length, # max. accepted token length
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size) 
        # what is nn.Embedding and what are the two embeddings for???
        # nn.Embedding maps each token in the input sequence (with vocab size 'src_vocab_size')
        # to a dense vector of size 'embed_size', same applies to self.position_embedding
        
        self.position_embedding = nn.Embedding(max_length, embed_size)
        # here the position embedding will be LEARNED during training, 
        # the original paper used fixed sinusoidal functions
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, 
                    heads, 
                    dropout=dropout, 
                    forward_expansion=forward_expansion,
                ) for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask): # x being the input
        N, seq_length = x.shape # N is the total number of input sequences
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # resulting in position tensor 
        # [[0, 1, 2, ..., seq_length - 1],
        #  [0, 1, 2, ..., seq_length - 1],
        #  ...
        #  [0, 1, 2, ..., seq_length - 1]] with shape N by seq_length
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module): # Decoder
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask) 
        # automatically calls TransformerBlock.forward() cos we used it as Function and passed in Tensors
        return out

class Decoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
             for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        
        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(
            self, 
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_size=256,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device="cuda",
            max_length=100,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask shape: (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
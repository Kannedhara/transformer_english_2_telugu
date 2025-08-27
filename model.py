import torch
import torch.nn as nn
import math

class EmbeddingBlock(nn.Module):

    def __init__(self, input_words_size=8000, emb_dim=512):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_embeddings=input_words_size, embedding_dim=emb_dim)

    
    def forward(self, x):
        ####print(x)
        ###print("embedding block input size is: ", x.shape)
        return self.embedding(x) * math.sqrt(self.emb_dim)
    


class PositionEncodingBlock(nn.Module):

        def __init__(self, seq, d_model=512):
            super().__init__()


            position = torch.arange(0, seq).unsqueeze(1)
            pe = torch.zeros(seq, d_model)
            self.der_term = torch.exp(torch.arange(0, d_model, 2)  * - (torch.log(torch.tensor(10000.0))/ d_model))

            pe[:, 0::2] = torch.sin(position * self.der_term)
            pe[:, 1::2] = torch.cos(position * self.der_term)

            self.register_buffer('pe', pe.unsqueeze(0)) 

        def forward(self, x):
            ###print("size of x", x.shape)
            ###print("size of pe", self.pe.shape)
            x = x + self.pe[:, :x.size(1)].detach()

            return x


class LayerNormalizationBlock(nn.Module):
     def __init__(self, eps: float = 10**-6):
          super().__init__()
          self.eps = eps
          self.gamma = nn.Parameter(torch.ones(1))
          self.beta = nn.Parameter(torch.zeros(1))
     
     def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


          
class FeedForwardBlock(nn.Module):
     
    def __init__(self, d_model):
        super().__init__()
        self.layer1 = nn.Linear(d_model, 2048)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(2048, d_model)

    def forward(self, x):
        x = self.layer2(self.relu1(self.layer1(x)))
        return x
     


class MultiAttentionBlock(nn.Module):
     
    def __init__(self, seq, d_model, h, dropout=0.1):
          super().__init__()

          self.seq = seq
          self.d_model = d_model
          self.h = h
          self.dropout = nn.Dropout(dropout)

          assert d_model % h == 0, "d_model must be divisible by number of heads"
          self.d_k = d_model // h


          self.w_q = nn.Linear(d_model, d_model)
          self.w_k = nn.Linear(d_model, d_model)
          self.w_v = nn.Linear(d_model, d_model)

          self.w_o = nn.Linear(d_model, d_model)


    @staticmethod
    def self_attention(q, k, v, mask, d_k):
         
         attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
         ###print("attention_scores size:", attention_scores.shape)
         ###print("mask size is:", mask.shape)
         
         if mask is not None:
              attention_scores.masked_fill_(mask == 0 , -1e9)

         attention_scores = attention_scores.softmax(dim = -1)


         return (attention_scores @ v), attention_scores
              
              
         
         

    def forward(self, q, k, v, mask):
       q = q.to(dtype=torch.float)
       k = k.to(dtype=torch.float)
       v = v.to(dtype=torch.float)
       ###print("q shape: ", q.shape)
       ###print("k shape: ", k.shape)
       ###print("v shape: ", v.shape)
       self.q = self.w_q(q)
       self.k = self.w_k(k)
       self.v = self.w_v(v)
     
       ###print("mask size in multi attenstion block forward : ", mask.shape) 
       self.q = self.q.view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2)
       self.k = self.k.view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)
       self.v = self.v.view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2)

       x, attention_scores = MultiAttentionBlock.self_attention(self.q, self.k, self.v, mask, self.d_k)


       x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

       return self.w_o(x)


class ResidualBlock(nn.Module):

     def __init__(self, dropout=0.1):
          super().__init__()
          self.layer_norm = LayerNormalizationBlock()
          self.dropout = nn.Dropout(dropout)

     def forward(self, x, sublayer, *args):
          x = x +  self.dropout(sublayer(self.layer_norm(x)))
          return x



class EncoderBlock(nn.Module):
     
    def __init__(self, self_attention_block: MultiAttentionBlock, feed_forward_block: FeedForwardBlock, dropout=0.1):
          super().__init__()
          self.self_attention_block = self_attention_block
          self.feed_forward_block = feed_forward_block
          self.residual_block = nn.ModuleList([ResidualBlock() for _ in range(2)])
          self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask):
         self.self_attention = lambda x: self.self_attention_block(x, x, x, mask)
         x = self.residual_block[0](x, self.self_attention)
         x = self.residual_block[1](x, self.feed_forward_block)
         return x



class  Encoder(nn.Module):
     
    def __init__(self, layers):
          super().__init__()

          self.layers = layers
          self.norm = LayerNormalizationBlock()

    def forward(self, x, mask):
         
        for layer in self.layers:
              x = layer(x, mask)

        return self.norm(x)
    


class DecoderBlock(nn.Module):
     
    def __init__(self, self_attention_block:MultiAttentionBlock, 
                        multi_attention_block: MultiAttentionBlock,
                        feed_forward_block: FeedForwardBlock,
                        dropout=0.1):
          super().__init__()

          self.self_attention_block = self_attention_block
          self.feed_forward_block = feed_forward_block
          self.multi_attention_block = multi_attention_block
          self.residual_block = nn.ModuleList([ResidualBlock() for _ in range(3)])
          self.dropout = nn.Dropout(dropout)
         

    def forward(self, x, memory, src_mask, tgt_mask):
         ###print("src_mask size in decoder : ", src_mask.shape) 
         ###print("tgt_mask size in decoder : ", tgt_mask.shape) 
         self.self_attention = lambda x: self.self_attention_block(x, x, x, src_mask)
         self.multi_attention = lambda x: self.multi_attention_block(memory, x, x, tgt_mask)
         x = self.residual_block[0](x, self.self_attention)
         x = self.residual_block[1](x, self.multi_attention)
         x = self.residual_block[2](x, self.feed_forward_block)

         return self.dropout(x)
    


class  Decoder(nn.Module):
     
    def __init__(self, layers):
          super().__init__()

          self.layers = layers
          self.norm = LayerNormalizationBlock()

    def forward(self, x, memory, src_mask, tgt_mask):
         
        ###print("src_mask size in decoder : ", src_mask.shape) 
        ###print("tgt_mask size in decoder : ", tgt_mask.shape) 
        for layer in self.layers:
              x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)
     
         

class ProjectionBlock(nn.Module):
     
    def __init__(self, d_model, vocab_size):
          super().__init__()

          self.layers = nn.Sequential(
          nn.Linear(d_model, 
                    d_model),
          nn.ReLU(),
          nn.LayerNorm(d_model),
          nn.Linear(d_model, vocab_size)
          )


    def forward(self, x):
        x = self.layers(x)
        return x
     
     
    

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder,
                  decoder: Decoder,
                  src_embed: EmbeddingBlock,
                  tgt_embed: EmbeddingBlock,
                  src_pos_encode: PositionEncodingBlock,
                  tgt_pos_encode: PositionEncodingBlock,
                  d_model,
                  vocab_size,
                  projection_layer: ProjectionBlock):
          super().__init__()

          self.encoder = encoder
          self.decoder = decoder
          self.src_embed = src_embed
          self.tgt_embed = tgt_embed
          self.src_pos_encode = src_pos_encode
          self.tgt_pos_encode = tgt_pos_encode
          self.d_model = d_model
          self.vocab_size = vocab_size
          self.projection_layer = projection_layer

    def encode(self, x, src_mask):    
         x = x.to(dtype=torch.long)
         x_embed = self.src_embed(x)
         ###print("x_embed shape:", x_embed.shape)
         x_pos = self.src_pos_encode(x_embed)
         x = self.encoder(x_pos, src_mask)
         return x
    
    def decode(self, x, encoder_output, src_mask, tgt_mask):
        x = x.to(dtype=torch.long)
        x_embed = self.tgt_embed(x)
        ###print("src_mask size in decode function is", src_mask.shape)
        ###print("tgt_mask size in decode function is", tgt_mask.shape)
        ###print("tgt_embed shape:", x_embed.shape)
        x = self.tgt_pos_encode(x_embed)
        x = self.decoder(x, encoder_output, src_mask, tgt_mask)
        return x

         

    def projection(self, x):
         return self.projection_layer(x)
    

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.projection_layer(output)




def build_transformer(
          src_vocab_size,
          tgt_vocab_size,
          src_seq_len,
          tgt_seq_len,
          N,
          d_model=512,
          h=8
):
     
     ###print("source vocab size", src_vocab_size)
     ###print("target vocab size", tgt_vocab_size)
     src_embed = EmbeddingBlock(src_vocab_size, d_model)
     tgt_embed = EmbeddingBlock(tgt_vocab_size, d_model)

     src_pos_encode = PositionEncodingBlock(src_seq_len, d_model)
     tgt_pos_encode = PositionEncodingBlock(tgt_seq_len, d_model)

     encoders = []

     for _ in range(N):
          self_attn_block = MultiAttentionBlock(seq=src_seq_len, d_model=d_model, h=h)
          feed_fwd_block = FeedForwardBlock(d_model=d_model)
          encoder_block = EncoderBlock(self_attention_block=self_attn_block, feed_forward_block=feed_fwd_block)
          encoders.append(encoder_block)
    

     decoders = []

     for _ in range(N):
          self_attn_block = MultiAttentionBlock(seq=src_seq_len, d_model=d_model, h=h)
          multi_attn_block = MultiAttentionBlock(seq=tgt_seq_len, d_model=d_model, h=h)
          feed_fwd_block = FeedForwardBlock(d_model=d_model)
          decoder_block = DecoderBlock(self_attention_block=self_attn_block,
                                       multi_attention_block=multi_attn_block,
                                       feed_forward_block=feed_fwd_block)
          decoders.append(decoder_block)

     encoder = Encoder(nn.ModuleList(encoders))
     decoder = Decoder(nn.ModuleList(decoders))


     projection_layer = ProjectionBlock(d_model, tgt_vocab_size)

     transformer = Transformer(
        encoder, decoder,
        src_embed, tgt_embed,
        src_pos_encode, tgt_pos_encode,
        d_model, tgt_vocab_size,
        projection_layer
        )

     

     for p in transformer.parameters():
          if p.dim() > 1:
               nn.init.xavier_uniform_(p)
    
     return transformer
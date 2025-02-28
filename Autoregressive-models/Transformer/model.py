import torch 
from torch.nn import functional as F
from torch import nn

import math 



class LayerNorm(nn.Module) : 

    def __init__(self,embed_dim,ep=1e-5) : 

        """
        defines LayerNormalizer.

        Args : 
            embed_dim (int) : determine size of embedding . 
        """
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim)) 
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.ep = ep

    def forward(self,x) : 

        """
        applies LayerNormalization to x.

        Returns (torch.Tensor) : Layer Normalized tensor.
        """

        mu = torch.mean(x,axis=-1).unsqueeze(-1)
        var = torch.var(x,axis=-1,unbiased=False).unsqueeze(-1)
        normalized_x = (x-mu)/torch.sqrt(var+self.ep)
        output = self.gamma * normalized_x + self.beta
        return output
    
class InputEmbedding(nn.Module) : 

    """
    Calculates a continuous representation for each token in the vocabulary using embeddings.

    This function generates embeddings that transform BPE encoded categorical token representations into dense vectors.

    Args:
        vocab_size (int): The number of unique tokens in the vocabulary. Determines the number of rows in the embedding matrix.
        d_model (int): The dimensionality of the embedding vectors. Each token will be represented as a vector of this size.
    """

    def __init__(self,vocab_size,embed_dim) : 
        super(InputEmbedding, self).__init__()
        self.d_model = embed_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
    
    def forward(self,x) : 
        """
        computes input embedding for certain sequence of vocalbularies.
        
        Args : 
            x (Torch.tensor) : sequence of tokens.

        Returns : 
            (Torch.tensor) : embedded representation of input sequence of shape (batch_size, seq_len, embedding_dim).
        """
        return self.embedding(x) * math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):

    """
    Computes postional embedding vectores for embedded sequence
    
    Args : 
        embed_dim : dimension of embedding
        seq_len :  length of sequence

    """

    def __init__(self, embed_dim, seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        position = torch.arange(0, seq_len).unsqueeze(1).float()  
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))

        pe = torch.zeros(seq_len, embed_dim) 
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        # Add a batch dimension and register as a buffer
        pe = pe.unsqueeze(0)  # Shape: (1, seq_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds up positional embedding vectore to input sequence.

        Returns (Torch.tensor) : positional embedded representation of input sequence of shape (batch_size, seq_len, embedding_dim).
        """
        return x + self.pe[:, :x.shape[1], :].requires_grad_(False)


class MultiHeadAttention(nn.Module) : 


    def __init__(self,embed_dim,seq_len,num_heads,masked=False):

        """
        Multi Head Attention Mechanism
        Args : 
        embed_dim (int) : dimension of embeded representation of each element in sequence.
        seq_len (int) : length of sequence
        num_heads (int) : number of attention heads
        """
        super().__init__()
        self.d_model ,self.seq_len = embed_dim , seq_len
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads
        self.masked = masked
        self.Q_proj = nn.Linear(self.d_model,self.d_model)
        self.K_proj = nn.Linear(self.d_model,self.d_model)
        self.V_proj = nn.Linear(self.d_model,self.d_model)
        self.Out_proj = nn.Linear(self.d_model,self.d_model)
        if masked : 
            mask = torch.tril(torch.ones(seq_len,seq_len)).float()
            self.register_buffer('mask',mask)


    def LinearTransform(self,K,Q,V) : 

        """
        Applies Linear transformation on key,query and value matrices

        Args : 

        K (torch.tensor) : Key 
        Q (torch.tensor) : Query 
        V (torch.tensor) : Value

        Returns : 
            Linear projected Key, Query and value tensor.
        """

        K = self.K_proj(K)
        Q = self.Q_proj(Q)
        V = self.V_proj(V)
        return K,Q,V

    def Attention(self,K,Q,V) : 

        """
        Applies Attention Mechanism on projected Key, Query and Value matrices
        
        Returns : 
            attention heads (torch.tensor) : tensor of shape (batch_size,num_heads,head_dim) containing attention heads.
            attention table (torch.tensor) : tensor of shape (batch_size,num_heads,seq_len,seq_len)
            
        """
        attention_table = torch.matmul(Q,K.permute(0,1,3,2))/math.sqrt(self.head_dim) #(batch_size,num_heads,seq_length,seq_length)
        attention_table = F.softmax(attention_table,dim=-1)
        if self.masked : 
            attention_table = attention_table * self.mask
        attention_heads = torch.matmul(attention_table,V)

        return attention_heads,attention_table
    

    def MultiHeadAttention(self,K,Q,V) : 

        """
        K : Tensor of shape : (batch_size,seq_len,d_model)
        Q : Tensor of shape : (batch_size,seq_len,d_model)
        V : Tensor of shape : (batch_size,seq_len,d_model)
        
        Return : 
            Attention table of shape : (batch_size,seq_length,seq_length)
            Concatenated Attention heads of shape : (batch_size,seq_length,d_model)
        """

        batch_size = K.shape[0]
        Q_split = Q.permute(0,2,1).view(batch_size,self.num_heads,self.head_dim,self.seq_len).permute(0,1,3,2) #(batch_size,num_heads,seq_length,head_dim)
        K_split = K.permute(0,2,1).view(batch_size,self.num_heads,self.head_dim,self.seq_len).permute(0,1,3,2) 
        V_split = V.permute(0,2,1).view(batch_size,self.num_heads,self.head_dim,self.seq_len).permute(0,1,3,2)
        attention_heads , attention_table = self.Attention(K_split,Q_split,V_split)
        attention_heads = attention_heads.permute(0,2,1,3).reshape(batch_size,self.seq_len,self.num_heads*self.head_dim)
        return attention_heads,attention_table
        

    def forward(self,K,Q,V) : 
        
        """
        takes K,Q,V and applies MultiHeadAttention mechanism.
        
        Returns : 
            attention heads (torch.tensor) : tensor of shape (batch_size,seq_length,d_model) containing attention heads.
            attention table (torch.tensor) : tensor of shape (batch_size,num_heads,seq_len,seq_len)
        """

        K,Q,V = self.LinearTransform(K,Q,V)
        attention_head , attention_table = self.MultiHeadAttention(K,Q,V)
        attention_head = self.Out_proj(attention_head)
        return attention_head,attention_table
    

class Encoder(nn.Module) :

    """
    Encoder part of transformers.
    """

    def __init__(self,embed_dim,seq_len,num_heads,vocab_size) : 

        """
            Encoder part of transformers.
            Args : 
                embed_dim (int) : embedding dimension of the input sequence.
                seq_len (int) : length of input sequence.
                num_heads (int) : number of attention heads.
                vocab_size (int) : maximum number of tokens in vocabulary.
        """
        super(Encoder,self).__init__()
        self.InputEmbedder = InputEmbedding(vocab_size=vocab_size,embed_dim=embed_dim)
        self.PosEmbedder = PositionalEncoding(embed_dim=embed_dim,seq_len=seq_len)
        self.MultiHeadAttention = MultiHeadAttention(embed_dim=embed_dim,seq_len=seq_len,num_heads=num_heads)
        self.FeedForward = nn.Linear(seq_len,seq_len)
        self.LayerNormalizerMultiHead = LayerNorm(embed_dim)
        self.LayerNormalizerOut = LayerNorm(embed_dim)

    def forward(self,x) : 
        """
            Computes operations on single layer of Encoder in transformer.
            Args : 
                x (torch.Tensor) : input sequence of shape (batch_size,1,seq_len) 
            
            Returns : 
                torch.Tensor (batch_size,seq_length,embed_dim) 

        """
        input_embedded = self.PosEmbedder(self.InputEmbedder(x))
        x = self.MultiHeadAttention(K=input_embedded,Q=input_embedded,V=input_embedded)
        AttentionNormalized = self.LayerNormalizerMultiHead(input_embedded + x)
        x = self.FeedForward(AttentionNormalized)
        x = F.relu(x)
        x = self.LayerNormalizerOut(x + AttentionNormalized)
        return x


class Decoder(nn.Module) : 

    def __init__(self,embed_dim,seq_len,num_heads,vocab_size) : 
        super(Decoder,self).__init__()
        self.InputEmbedder = InputEmbedding(vocab_size=vocab_size,embed_dim=embed_dim)
        self.PosEmbedder = PositionalEncoding(embed_dim=embed_dim,seq_len=seq_len)
        self.MaskedMultiHeadAttention = MultiHeadAttention(embed_dim=embed_dim,seq_len=seq_len,num_heads=num_heads,masked=True)
        self.MultiHeadAttention= MultiHeadAttention(embed_dim=embed_dim,seq_len=seq_len,num_heads=num_heads)
        self.LayerNormalizerMaskedAttention = LayerNorm(embed_dim)
        self.LayerNormalizerCrossAttention = LayerNorm(embed_dim)
        self.LayerNormalizerOut = LayerNorm(embed_dim)
        self.FeedForward = nn.Linear(embed_dim,embed_dim)
    
    def forward(self,x,EncoderOut='self') : 
        input_embedded = self.InputEmbedder(x)
        input_embedded = self.PosEmbedder(input_embedded)
        x , _ = self.MultiHeadAttention(input_embedded,input_embedded,input_embedded)
        AttentionNormalized = self.LayerNormalizerMaskedAttention(input_embedded + x)
        if EncoderOut == 'self' : 
            CrossAttention, _ = self.MultiHeadAttention(AttentionNormalized,AttentionNormalized,AttentionNormalized)
        CrossAttentionNormalized = self.LayerNormalizerCrossAttention(CrossAttention + AttentionNormalized)
        x = self.FeedForward(CrossAttentionNormalized)
        x = F.relu(x)
        x = self.LayerNormalizerOut(x + CrossAttentionNormalized)
        return x



class NGram(nn.Module) : 

    def __init__(self,embed_dim,seq_len,num_heads,vocab_size) :
        self.Decoder = Decoder(embed_dim=embed_dim,seq_len=seq_len,num_heads=num_heads,vocab_size=vocab_size)
        self.linear = nn.Linear(embed_dim,embed_dim)
    
    def forward(self,x) : 
        x = self.Decoder(x)
        x = self.linear(x) #(batch_size,seq_length,vocab_size)
        x = F.log_softmax(x,dim=-1)
        return x
    


        

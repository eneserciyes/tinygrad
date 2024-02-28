from tinygrad import Tensor,nn

class MultiHeadAttention:
  def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
    self.n_head = n_head
    self.d_k = d_k
    self.d_v = d_v
    self.dropout = dropout

    self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
    self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
    self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
    self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

  def __call__(self, q, k, v, mask=None):
    d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
    sz_b, len_q, len_k, len_v = q.shape[0], q.shape[1], k.shape[1], v.shape[1]

    residual = q
    q = self.w_qs(q).reshape(sz_b, len_q, n_head, d_k)
    k = self.w_ks(k).reshape(sz_b, len_k, n_head, d_k)
    v = self.w_vs(v).reshape(sz_b, len_v, n_head, d_v)

    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    if mask is not None: mask = mask.unsqueeze(1)

    q = Tensor.scaled_dot_product_attention(q, k, v, mask)

    q = q.transpose(1, 2).contiguous().reshape(sz_b, len_q, -1)
    q = self.fc(q).dropout(self.dropout) + residual

    return self.layer_norm(q)

class PositionwiseFeedForward:
  def __init__(self, d_in, d_hid, dropout=0.1):
    self.w_1 = nn.Linear(d_in, d_hid) 
    self.w_2 = nn.Linear(d_hid, d_in)
    self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
    self.dropout = dropout

  def __call__(self, x):
    x = x + self.w_2(self.w_1(x).relu()).dropout(self.dropout)
    return self.layer_norm(x) 

class AttentionBlock:
  def __init__(self, feat_dim, hidden_dim, num_heads, dropout):
    self.slf_attn = MultiHeadAttention(num_heads, feat_dim, feat_dim//num_heads, feat_dim//num_heads, dropout=dropout)
    self.pos_ffn = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

  def __call__(self, enc_input, slf_attn_mask=None):
    enc_output = self.slf_attn(enc_input, enc_input, enc_input, slf_attn_mask)
    enc_output = self.pos_ffn(enc_output)
    return enc_output.realize()

class AttentionBlockKVCache:
  def __init__(self, feat_dim, hidden_dim, num_heads, dropout):
    self.slf_attn = MultiHeadAttention(num_heads, feat_dim, feat_dim//num_heads, feat_dim//num_heads, dropout=dropout)
    self.pos_ffn = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

  def __call__(self, q, k, v, slf_attn_mask=None):
    enc_output = self.slf_attn(q, k, v, slf_attn_mask)
    enc_output = self.pos_ffn(enc_output)
    return enc_output.realize()

class PositionalEncoding1D:
  def __init__(self, max_length: int, embed_dim: int):
    self.max_length = max_length
    self.embed_dim = embed_dim
    self.pos_emb = nn.Embedding(max_length, embed_dim)

  def __call__(self, feat):
    pos_emb = self.pos_emb(Tensor.arange(self.max_length).reshape(1, -1))
    pos_emb = pos_emb.repeat((feat.shape[0], 1, 1))
    return feat + pos_emb[:, :feat.shape[1], :]

  def forward_with_position(self, feat, position):
    assert feat.shape[1] == 1
    pos_emb = self.pos_emb(Tensor.arange(self.max_length).unsqueeze(0).repeat((feat.shape[0], 1)))
    return feat + pos_emb[:, position:position+1, :]


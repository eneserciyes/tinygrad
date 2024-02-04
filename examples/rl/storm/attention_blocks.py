from tinygrad import Tensor,nn

class MultiHeadAttention:
  def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
    self.n_head = n_head
    self.d_k = d_k
    self.d_v = d_v
    self.dropout = dropout

    self.query = nn.Linear(d_model, n_head * d_k, bias=False)
    self.key = nn.Linear(d_model, n_head * d_k, bias=False)
    self.value = nn.Linear(d_model, n_head * d_v, bias=False)
    self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

  def __call__(self, q, k, v, mask=None):
    d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
    bs, qs, ks, vs = q.shape[0], q.shape[1], k.shape[1], v.shape[1]

    residual = q
    q = self.query(q).reshape(bs, qs, n_head, d_k)
    k = self.key(k).reshape(bs, ks, n_head, d_k)
    v = self.value(v).reshape(bs, vs, n_head, d_v)

    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    if mask is not None: mask = mask.unsqueeze(1)

    q = Tensor.scaled_dot_product_attention(q, k, v, mask)

    q = q.transpose(1, 2).contiguous().reshape(bs, qs, -1)
    q = self.fc(q).dropout(self.dropout) + residual

    return q.layer_norm()

class PositionwiseFeedForward:
  def __init__(self, d_in, d_hid, dropout=0.1):
    self.fc1 = (Tensor.scaled_uniform(d_in, d_hid), Tensor.zeros(d_hid))
    self.fc2 = (Tensor.scaled_uniform(d_hid, d_in), Tensor.zeros(d_in))
    self.dropout = dropout

  def __call__(self, x):
    x = x + x.linear(*self.fc1).relu().linear(*self.fc2).dropout(self.dropout)
    return x.layer_norm()

class AttentionBlock:
  def __init__(self, feat_dim, hidden_dim, num_heads, dropout):
    self.slf_attn = MultiHeadAttention(num_heads, feat_dim, feat_dim//num_heads, feat_dim//num_heads, dropout=dropout)
    self.pos_ffn = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

  def __call__(self, enc_input, slf_attn_mask=None):
    enc_output = self.slf_attn(enc_input, enc_input, enc_input, slf_attn_mask)
    enc_output = self.pos_ffn(enc_output)
    return enc_output

class AttentionBlockKVCache:
  def __init__(self, feat_dim, hidden_dim, num_heads, dropout):
    self.slf_attn = MultiHeadAttention(num_heads, feat_dim, feat_dim//num_heads, feat_dim//num_heads, dropout=dropout)
    self.pos_ffn = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

  def __call__(self, q, k, v, slf_attn_mask=None):
    enc_output = self.slf_attn(q, k, v, slf_attn_mask)
    enc_output = self.pos_ffn(enc_output)
    return enc_output

class PositionalEncoding1D:
  def __init__(self, max_length: int, embed_dim: int):
    self.max_length = max_length
    self.embed_dim = embed_dim
    self.pos_emb = nn.Embedding(max_length, embed_dim)

  def __call__(self, feat):
    pos_emb = self.pos_emb(Tensor.arange(self.max_length))
    pos_emb = pos_emb.unsqueeze(0).repeat((feat.shape[0], 1, 1))
    return feat + pos_emb[:, :feat.shape[1], :]

  def forward_with_position(self, feat, position):
    assert feat.shape[1] == 1
    pos_emb = self.pos_emb(Tensor.arange(self.max_length))
    pos_emb = pos_emb.unsqueeze(0).repeat((feat.shape[0], 1, 1))
    return feat + pos_emb[:, position:position+1, :]


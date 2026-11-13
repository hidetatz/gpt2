import numpy as np

class CharTokenizer:
    # Although the real GPT-2 uses sub-word tokenization, this implementation
    # uses the simpler one.
    def __init__(self, corpus: str):
        # create raw char -> token_ID dictionary from the given corpus
        self.chars = sorted(set(corpus))
        self.char_to_token = {c: i for i, c in enumerate(self.chars)}
        self.token_to_char = {i: c for c, i in self.char_to_token.items()}
        self.vocab_size = len(self.chars)

    # text -> tokens
    def encode(self, text):
        return np.array([self.char_to_token[c] for c in text], dtype=np.int64)

    # tokens -> text
    def decode(self, tokens):
        return "".join([self.token_to_char[int(t)] for t in tokens])

# create positional embedding in advance to tell the token position.
# In GPT-2 this is the trained parameter, but the Attention is all you need paper it's fixed.
# In this implementation the original sin/cos calculation is used.
# While "pos" is the token position in the input,
# PE(pos, 2i)   = sin(pos / (10000^(2i/D)))
# PE(pos, 2i+1) = cos(pos / (10000^(2i/D)))
def pos_emb(input_tokens_limit, model_d):
    pos = np.arange(input_tokens_limit).reshape(input_tokens_limit, 1)  # (1024, 1)
    i_mul_2 = np.arange(0, model_d, 2).reshape(1, model_d//2)  # This is 2i (0, 2, 4, ... 62), (1, 32)
    angle = pos / np.power(10000.0, i_mul_2 / model_d)
    pe = np.zeros((input_tokens_limit, model_d)) # (1024, 64)
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

# configs
# in the real GPT-2 small, the parameters are:
# vocab_size: 50257
# input_tokens_limit: 1024
# d_model: 768
# n_heads: 12
# d_ff: 3072 (768 * 4)
# n_layers: 12
input_tokens_limit = 1024
d_model = 64
n_heads = 4 # each head processes 16 (64 / 4) dims
d_ff = 256 # 64 * 4
n_layers = 2

assert d_model % n_heads == 0

rng = np.random.default_rng(42)

class LayerNorm:
    def __init__(self):
        # trained parameters: uses 1 in this implementation
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = 1e-5

    def __call__(self, x): # x: (T, d_model)
        # LayerNorm: normalizes the each token vector with mean = 0 and variance = 1,
        # then multiplies scale (gamma) and shift (beta).
        # This is needed to prevent x explode/vanish.
        # LN(x) = gamma * (x - mean) / sqrt(variance + eps) + beta
        # does not change x shape
        mean = x.mean(axis=-1, keepdims=True) # (T, 1)
        var = x.var(axis=-1, keepdims=True) # (T, 1)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta # (T, d_model) -> (T, d_model)

class MultiHeadAttention:
    def __init__(self):
        self.h = n_heads
        self.d_k = d_model // n_heads
        s = 1 / np.sqrt(d_model) # to prevent value size changes a lot after matmul

        # trained parameters
        # attn.c_attn
        self.W_attn = rng.normal(0, s, (d_model, d_model*3))
        self.b_attn = np.zeros(d_model*3)
        # attn.c_proj
        self.W_proj = rng.normal(0, s, (d_model, d_model))
        self.b_proj = np.zeros(d_model)

    def __call__(self, x, mask): # x: (T, 64), mask: (T, T)
        T = x.shape[0]

        # make Q, K, V
        qkv = x @ self.W_attn + self.b_attn # (T, 64) @ (64, 192) -> (T, 192)
        qkv = qkv.reshape(T, 3, self.h, self.d_k) # (T, 192) -> (T, 3, 4, 16)
        qkv = qkv.transpose(1, 2, 0, 3) # (T, 3, 4, 16) -> (3, 4, T, 16)
        Q, K, V = qkv[0], qkv[1], qkv[2] # (4, T, 16)

        # scaled dot product attention
        # multiplies Q and K then apply causal mask
        scores = Q @ K.swapaxes(-1, -2) / np.sqrt(self.d_k) # (4, T, 16) @ (4, 16, T) -> (4, T, T)
        scores = np.where(mask, -1e9, scores) # mask is broadcasted into (4, T, T)

        # softmax
        weights = softmax(scores) # (4, T, T) -> (4, T, T)

        # get V
        out = weights @ V # (4, T, T) * (4, T, 16) -> (4, T, 16)

        # combine heads output
        out = out.transpose(1, 0, 2) # (4, T, 16) -> (T, 4, 16)
        out = out.reshape(T, self.h * self.d_k) # (T, 4, 16) -> (T, 64)

        # apply output projection
        return out @ self.W_proj + self.b_proj

class FeedForwardNetwork:
    def __init__(self):
        self.w1 = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_ff)) # (64, 256)
        self.b1 = np.zeros(d_ff) # (256,)
        self.w2 = rng.normal(0, 1 / np.sqrt(d_ff), (d_ff, d_model)) # (256, 64)
        self.b2 = np.zeros(d_model) # (64,)

    def __call__(self, x): # x: (T, 64)
        # expand the shape for non-linear conversion
        x = x @ self.w1 + self.b1 # (T, 64) @ (64, 256) -> (T, 256)

        # gelu, GPT2 uses tanh approximation
        # GELU_tanh(x) = 0.5x(1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

        # recover the shape
        x = x @ self.w2 + self.b2 # (T, 256) @ (256, 64) -> (T, 64)
        return x

class TransformerBlock:
    def __init__(self):
        self.ln_1 = LayerNorm()
        self.mha = MultiHeadAttention()
        self.ln_2 = LayerNorm()
        self.ffn = FeedForwardNetwork()

    def __call__(self, x, mask):
        # Each transformer block just does 4 things
        # * apply first layer norm
        # * do residual connection x with multi_head_attention(x, causal_mask)
        # * apply second layer norm
        # * do residual connection x with feed_forward_network(x)
        # In the GPT-2 there are several transformer blocks but it's done in GPT2 class __call__ method.
        x = x + self.mha(self.ln_1(x), mask) # (T, 64) -> (T, 64)
        x = x + self.ffn(self.ln_2(x))       # (T, 64) -> (T, 64)
        return x

class GPT2:
    def __init__(self, vocab_size):
        # trained params: token embedding and positional embedding
        # uses random in this implementation
        self.wte = rng.normal(0, 0.02, (vocab_size, d_model))
        self.wpe = pos_emb(input_tokens_limit, d_model)
        self.blocks = [TransformerBlock() for _ in range(n_layers)]
        self.ln_f = LayerNorm()

    def __call__(self, tokens):
        T = len(tokens)  # input tokens size

        # apply token embedding and positional encoding
        # multiplies sqrt(d_model) as the original paper
        x = self.wte[tokens] * np.sqrt(d_model) + self.wpe[:T] # (T, 64)

        # make causal mask: the upper right gets True (= masked) to prevent leak
        causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)

        # apply transformers
        for block in self.blocks: x = block(x, causal_mask) # (T, 64) -> (T, 64)

        # final layer norm
        x = self.ln_f(x) # (T, 64) -> (T, 64)

        # This is the final result logit
        return x @ self.wte.T # (T, 64) @ (64, vocab_size) -> (T, vocab_size)

# self recursive generation
def generate(model, tokens, n_new, temperature=0.0):
    tokens = list(tokens)
    for _ in range(n_new):
        context = np.array(tokens[-input_tokens_limit:]) # cut input not to exceed limit
        logits = model(context)[-1]  # the last token logits
        if temperature == 0.0:
            next_token = int(np.argmax(logits))
        else:
            probs = softmax(logits/temperature)
            next_token = int(rng.choice(len(probs), p=probs))
        tokens.append(next_token)
    return tokens

corpus = 'Generative Pre-trained Transformer 2 (GPT-2) is a large language model (LLM) by OpenAI and the second in their foundational series of GPT models. GPT-2 was pre-trained on a dataset of 8 million web pages. It was partially released in February 2019, followed by full release of the 1.5-billion-parameter model on November 5, 2019. GPT-2 was created as a "direct scale-up" of GPT-1 with a ten-fold increase in both its parameter count and the size of its training dataset. It is a general-purpose learner and its ability to perform the various tasks was a consequence of its general ability to accurately predict the next item in a sequence, which enabled it to translate texts, answer questions about a topic from a text, summarize passages from a larger text, and generate text output on a level sometimes indistinguishable from that of humans; however, it could become repetitive or nonsensical when generating long passages. It was superseded by the GPT-3 and GPT-4 models, which are no longer open source. GPT-2 has, like its predecessor GPT-1 and its successors GPT-3, GPT-4 and GPT-5, a generative pre-trained transformer architecture, implementing a deep neural network, specifically a transformer model, which uses attention instead of older recurrence- and convolution-based architectures. Attention mechanisms allow the model to selectively focus on segments of input text it predicts to be the most relevant. This model allows for greatly increased parallelization, and outperforms previous benchmarks for RNN/CNN/LSTM-based models.'

tok = CharTokenizer(corpus)
print(f"{tok.vocab_size=}")
model = GPT2(tok.vocab_size)

print(tok.decode(generate(model, tok.encode("hello "), 20, 1.0)))

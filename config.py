

class Config:

    def __init__(self):

        self.max_length = 16
        self.img_size = 300
        self.vocab_size = 5000
        self.embedding_dim = 512

        self.dropout_rate = 0.2
        self.ffn_dim = 512
        self.n_heads = 4
        self.n_layer = 8

       
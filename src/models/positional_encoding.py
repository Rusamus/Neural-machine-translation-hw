import torch

class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size, maxlen, device, n=10000):
        """
        emb_size - размер эмбеддингов
        maxlen - длинна контекста
        """
        super().__init__()
        # TODO: Реализуйте конструтор
        self.positional_encoding = torch.zeros((maxlen, emb_size), device=device)
        for k in range(maxlen):
            for i in torch.arange(int(emb_size/2)):
                denominator = n ** (2*i/emb_size)
                self.positional_encoding[k, 2*i] = torch.sin(k/denominator)
                self.positional_encoding[k, 2*i+1] = torch.cos(k/denominator)

    def forward(self, token_embedding):
        """
        token_embedding - тензор матрицы эмбеддингов
        """
        # TODO: Реализуйте сложение эмбединнгов токенов с позиционными эмбеддингами
        return token_embedding + self.positional_encoding
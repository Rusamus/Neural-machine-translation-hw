import torch
import numpy as np
import metrics
from models.positional_encoding import PositionalEncoding


class Seq2SeqTransformer(torch.nn.Module):
    def __init__(self, model_config, word2index):
        super().__init__()
       # TODO: Реализуйте конструктор seq2seq трансформера - матрица эмбеддингов, позиционные эмбеддинги, encoder/decoder трансформер, vocab projection head
        
        src_vocab_size = model_config['src_vocab_size']
        tgt_vocab_size = model_config['tgt_vocab_size']
        embedding_size = model_config['embedding_size']
        nhead = model_config['nhead']
        num_encoder_layers = model_config['num_encoder_layers']
        lr = model_config['lr']
        weight_decay = model_config['weight_decay']
        T_max = model_config['T_max']
        maxlen = model_config['maxlen']
        self.device = model_config['device']

        self.src_embed = torch.nn.Embedding(src_vocab_size, embedding_size)
        self.tgt_embed = torch.nn.Embedding(tgt_vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, maxlen, device=self.device)

        self.transformer_model = torch.nn.Transformer(
                d_model=embedding_size, nhead=nhead, 
                num_encoder_layers=num_encoder_layers
            )
        self.fc = torch.nn.Linear(embedding_size, tgt_vocab_size)
        self.criterion = torch.nn.CrossEntropyLoss()

        parameters = []
        parameters.append({'params': self.src_embed.parameters()})
        parameters.append({'params': self.tgt_embed.parameters()})
        parameters.append({'params': self.transformer_model.parameters()})
        parameters.append({'params': self.fc.parameters()})
        
        self.optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max)
        self.word2index = word2index



    def generate_tgt_msk(self, tgt):
        mask = (torch.triu(torch.ones(tgt.shape[1], tgt.shape[1], device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_src_msk(self, src):
        mask = torch.zeros((src.shape[1], src.shape[1]), 
            device=self.device).type(torch.bool)
        return mask
    
    def generate_all_masks(self, src, tgt):
        src_mask = self.generate_src_msk(src)
        tgt_mask = self.generate_tgt_msk(src)

        src_key_padding_mask = (src == self.word2index["[PAD]"]).type(torch.bool)
        tgt_key_padding_mask = (tgt == self.word2index["[PAD]"]).type(torch.bool)

        return src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None): 

        src_embed = self.src_embed(src) # B x block_size x embed_size
        src_embed = self.positional_encoding(src_embed) 
        tgt_embed = self.tgt_embed(tgt) # B x block_size x embed_size
        tgt_embed = self.positional_encoding(tgt_embed)

        out = self.transformer_model(src_embed, 
            tgt_embed, src_mask=src_mask, 
            tgt_mask=tgt_mask, memory_mask=None, 
            src_key_padding_mask=src_key_padding_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask)
        
        output = self.fc(out)

        return output


    def training_step(self, batch):
        
        input_tensor, target_tensor = batch
        self.optimizer.zero_grad()

        src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = self.generate_all_masks(input_tensor, target_tensor)
        decoder_outputs = self.forward(input_tensor, target_tensor, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        target_tensor = target_tensor[:, :, None]
        target_length = target_tensor.shape[1]
        loss = 0.0
        for di in range(target_length):
            loss += self.criterion(
                decoder_outputs[:, di, :].squeeze(), target_tensor[:, di, :].squeeze()
            )
        loss = loss / target_length
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def validation_step(self, batch):
        input_tensor, target_tensor = batch
        with torch.no_grad():
            src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = self.generate_all_masks(input_tensor, target_tensor)
            decoder_outputs = self.forward(input_tensor, target_tensor, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            target_tensor = target_tensor[:, :, None]
            target_length = target_tensor.shape[1]
            loss = 0
            for di in range(target_length):
                loss += self.criterion(
                    decoder_outputs[:, di, :].squeeze(), target_tensor[:, di, :].squeeze()
                )
            loss = loss / target_length

        return loss.item()


    def eval_bleu(self, predicted, target_tensor, target_tokenizer):
        predicted = predicted.squeeze().detach().cpu().numpy().astype(int)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy().astype(int)[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences
    
    def encode(self, src, src_mask, src_key_padding_mask=None):
        src_embed = self.src_embed(src)
        src_embed = self.positional_encoding(src_embed)
        encoded = self.transformer.encoder(src_embed, src_mask, src_key_padding_mask)

        return encoded

    def decode(self, tgt, memory, tgt_mask=None, src_mask=None, tgt_key_padding_mask=None,
        src_key_padding_mask=None):

        tgt_embed = self.tgt_embed(tgt)
        tgt_embed = self.positional_encoding(tgt_embed)

        decoded = self.transformer.decoder(
                tgt_embed, memory, tgt_mask, 
                src_mask, tgt_key_padding_mask,
                src_key_padding_mask,
        )
        return decoded


    def generate(self, src):

        src_mask = self.generate_src_mask()
        tgt_mask = self.generate_tgt_mask()

        encoded = self.encode(src, src_mask)
        idx = torch.zeros((src.shape[0], 1)) # batch x 1

        for _ in range(self.max_len):
            output = self.decode(idx, encoded, tgt_mask, src_mask)
            logits = self.fc(output)

            probs = torch.nn.functional.softmax(logits, dim=-1)

            # generate new word based on probs
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
 
        return idx
    

    # def forward(self, input_tensor: torch.Tensor):
    #     # TODO: Реализуйте forward pass для модели, при необходимости реализуйте другие функции для обучения
    #     output = self.src_embed(input_tensor)
    #     output = self.positional_encoding(output)
    #     output = self.transformer_model(output)
    #     output = self.softmax(self.fc(output))
    #     return output

    # def create_src_mask(self, src): 
    #     src_mask = (src != 0).unsqueeze(-2) 
    #     return src_mask

    # def create_tgt_mask(self, tgt): 
    #     tgt_pad_mask = (tgt != 0).unsqueeze(-2) 
    #     tgt_len = tgt.size(1) 
    #     tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len)).cuda() 
    #     tgt_mask = torch.where(tgt_sub_mask == 1, tgt_pad_mask, 0).bool()
    #     return tgt_mask


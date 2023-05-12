import torch
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration
from transformers.optimization import Adafactor
import metrics


class Seq2SeqT5(torch.nn.Module):
    def __init__(self, config, pretrained_model="t5-small"):
        super().__init__()
        self.device = config["device"]

        self.model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model).to(self.device)
        self.model.resize_token_embeddings(config["tokenizer_length"])

        self.optimizer = Adafactor(
            self.model.parameters(), lr=config['lr'], relative_step=False)
        self.criterion = CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_tensor: torch.Tensor, labels):
        output = self.model(input_ids=input_tensor, labels=labels)
        return output

    def training_step(self, batch):
        self.optimizer.zero_grad()
        input_tensor, target_tensor = batch
        loss = self.forward(input_tensor, target_tensor).loss
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validation_step(self, batch):
        input_tensor, target_tensor = batch
        with torch.no_grad():
            loss = self.forward(input_tensor, target_tensor).loss

        return loss.item()

    def eval_bleu(self, predicted, target_tensor, target_tokenizer=None):
        predicted = predicted.squeeze().detach().cpu().numpy().astype(int)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy().astype(int)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences

    def generate(self, src):
        outputs = self.model.generate(src)
        return outputs

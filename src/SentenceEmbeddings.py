##############
# ESPECIALLY DESIGNED FOR NOTEBOOKS V5 and upwards
##############

import torch
from transformers import BertTokenizer, BertModel, BertConfig


class SentenceEmbeddings:
    def __init__(self, model_id_keywords: str, device: str) -> None:
        self.tokenizer_sentence = BertTokenizer.from_pretrained(model_id_keywords)
        self.model_sentence = (
            BertModel.from_pretrained(model_id_keywords, output_hidden_states=True)
            .eval()
            .to(device)
        )
        self.sentence_dim = BertConfig.from_pretrained(model_id_keywords).hidden_size
        self.device = device

    def encode(
        self,
        sentences: list[str],
    ) -> torch.Tensor:
        """
        Encodes sentences using BERT.

        Args:
            sentences (list): list of strings to encode

        Returns:
            torch.Tensor: tensor of encoded sentences with dimension (len(sentences), self.sentence_dim)
        """
        encodings = []
        for sentence in sentences:
            # Lower because of the uncased model
            sentence = sentence.lower()

            # Add the special tokens.
            sentence = "[CLS] " + sentence + " [SEP]"

            indexed_tokens = self.tokenizer_sentence.encode(
                sentence, padding="max_length", truncation=True
            )
            # same as self.convert_tokens_to_ids(self.tokenize(text))

            # Mark each of the 3 OR 5 tokens as belonging to sentence "1".
            segments_ids = [1] * len(indexed_tokens)

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
            segments_tensors = torch.tensor([segments_ids]).to(self.device)

            # Use BERT to encode the sentence and use the sum of the last four layers.
            with torch.no_grad():
                outputs = self.model_sentence(tokens_tensor, segments_tensors)
                # print(outputs)
            hidden_states = outputs.last_hidden_state[0]
            # print(hidden_states.size())
            last_four = hidden_states[-4:]
            # print(last_four.size())
            summation = torch.sum(last_four, 0)

            encodings.append(summation)

        return torch.stack(encodings)

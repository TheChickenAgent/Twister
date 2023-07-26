import re
import torch
from itertools import combinations
from transformers import BertTokenizer, BertModel, BertConfig


class KeywordEmbeddings:
    """
    KeywordEmbeddings: class that generates keyword embeddings for each class in the dataset
    """

    def __init__(self, model_id_keywords: str, device: str) -> None:
        self.tokenizer_keywords = BertTokenizer.from_pretrained(model_id_keywords)
        self.model_keywords = (
            BertModel.from_pretrained(model_id_keywords, output_hidden_states=True)
            .eval()
            .to(device)
        )
        self.keyword_dim = BertConfig.from_pretrained(model_id_keywords).hidden_size
        self.device = device

    def sublabel_keywords(
        self, dfs: dict, keywords_same: bool, main_label: str, sub_label: str
    ) -> dict:
        """
        Uses the sublabel structure of the dataset to generate a dictionary of keywords for each split and for each class.
        For example: {train: {'Agent': ['King', ...]}}

        Args:
            dfs (dict): dataframes from to_dataframe
            keywords_same (bool): boolean to check if all sets of keywords between train/val/test splits are the same
            main_label (str): the upper label that we view from
            sub_label (str): the sub label that is included in the main_label

        Returns:
            dict: keywords through dataset set with main-sub structure
        """
        split_labels = {}

        for split in dfs:
            dataframe = (
                dfs[split][[main_label, sub_label]]
                .groupby([main_label, sub_label])
                .count()
                .reset_index()
            )
            main_labels = dataframe["l1"].to_list()

            label_keywords = {}
            split_labels[split] = label_keywords
            for label in main_labels:
                label_keywords[label] = dataframe[dataframe["l1"] == label][
                    "l2"
                ].to_list()

        if keywords_same:
            # Assert that the keywords occur in every split
            # First, make combinations between the keys, e.g. (train, test) and (train, validation), ...
            splits_combinations = list(combinations(list(dfs.keys()), 2))
            for split_combination in splits_combinations:
                split_1 = split_combination[0]
                split_2 = split_combination[1]
                assert set(split_labels[split_1].keys()) == set(
                    split_labels[split_2].keys()
                )

        return split_labels

    def encode_keywords(self, keywords: list):
        """
        Encodes each keyword in the list, uses mean pooling to construct an overall embedding

        Args:
            keywords (list): list of keywords in string format

        Returns:
            torch.Tensor: tensor representing the overall label embedding
        """
        keyword_embeddings = []

        for keyword in keywords:
            # Add the special tokens.
            text = "[CLS] " + keyword + " [SEP]"

            # Split the sentence into tokens.
            tokenized_text = self.tokenizer_keywords.tokenize(text)

            # Map the token strings to their vocabulary indeces.
            indexed_tokens = self.tokenizer_keywords.convert_tokens_to_ids(
                tokenized_text
            )

            # Mark each of the 3 OR 5 tokens as belonging to sentence "1".
            segments_ids = [1] * len(tokenized_text)

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
            segments_tensors = torch.tensor([segments_ids]).to(self.device)

            # Run the text through BERT, and collect all of the hidden states produced
            # from all 12 layers.
            with torch.no_grad():
                outputs = self.model_keywords(tokens_tensor, segments_tensors)
                hidden_states = outputs[2]

            token_vecs = hidden_states[-2][0]
            # print(token_vecs.size()) #[tokens, 768]

            # Calculate the average of all token vectors.
            average_token_emb = torch.mean(token_vecs, dim=0)
            keyword_embeddings.append(average_token_emb)
        keyword_embeddings = torch.stack(keyword_embeddings)

        keyword_embeddings = torch.mean(keyword_embeddings, dim=0)

        return keyword_embeddings

    def encoded_mapping(self, split_labels: dict) -> dict:
        """
        Maps the upper classes to an overall class embedding of the lower classes. ASSUMES all dataset splits contain the same keywords

        Args:
            split_labels (dict): dataset splits with inner dictionary representing the class labels and sub-labels

        Returns:
            dict: dictionary with classes and their average embedding based on the sub-keywords
        """
        l1_to_l2_encoded = {}

        for l1_class in split_labels["train"]:
            l2_keywords = split_labels["train"][l1_class]  # Access keywords in L2
            # print(l2_keywords)

            # Check their spelling, e.g. BritishRoyalty --> British Royalty --> british royalty
            for i in range(len(l2_keywords)):
                keyword = l2_keywords[i]
                keyword_spaced = re.sub(
                    r"(?<=[a-zA-Z])(?=[A-Z])", " ", keyword
                )  # replace camal case with space
                keyword = keyword.lower()  # make it lower case
                l2_keywords[
                    i
                ] = keyword_spaced  # replace the original keyword with the adjusted one
            # print(l2_keywords)

            l2_encoded = self.encode_keywords(
                l2_keywords
            )  # encode all keywords in one class of l1
            l1_to_l2_encoded[
                l1_class
            ] = l2_encoded  # put the encodings into the l1_to_l2_encoded

        return l1_to_l2_encoded

    def encoded_mapping_class_description(self, split_labels: dict) -> dict:
        """
        Maps the upper classes to an overall class embedding using a description of the classes. ASSUMES all dataset splits contain the same keywords

        Args:
            split_labels (dict): dataset splits with inner dictionary representing the class labels and descriptions

        Returns:
            dict: dictionary with classes and their average embedding based on the sub-keywords
        """
        l1_to_l2_encoded = {}

        for l1_class in split_labels:
            l2_description = split_labels[l1_class]  # Access keywords in L2
            l2_description = [l2_description.lower()]  # Make it lower case and a list
            # print(l2_keywords)

            l2_encoded = self.encode_keywords(
                l2_description
            )  # encode all keywords in one class of l1
            l1_to_l2_encoded[
                l1_class
            ] = l2_encoded  # put the encodings into the l1_to_l2_encoded

        return l1_to_l2_encoded

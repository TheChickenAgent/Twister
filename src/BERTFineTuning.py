import re
import time
import nltk
import torch
import random
import numpy as np
import datetime as dt

nltk.download("stopwords")

from nltk.corpus import stopwords
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchmetrics.classification import (
    MulticlassRecall,
    MulticlassPrecision,
    MulticlassAccuracy,
    MulticlassF1Score,
)


class BERTFineTuning:
    """
    BERTFineTuning: class for fine-tuning BERT models
    """

    def __init__(self, model_id: str, device: str, SEED: int = 42):
        self.tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained(
            model_id,
            num_labels=9,
            output_attentions=False,
            output_hidden_states=False,
        ).to(device)
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        self.SEED = SEED
        self.set_seed()

    def set_seed(self):
        """
        Set seed for reproducibility
        """
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed_all(self.SEED)

    def clean_text(self, text: str) -> str:
        """
        Cleans text for the BERT model

        Args:
            text (str): vanilla text

        Returns:
            str: cleaned text
        """
        sw = stopwords.words("english")

        text = text.lower()

        text = re.sub(
            r"[^a-zA-Z?.!,¿]+", " ", text
        )  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")

        html = re.compile(r"<.*?>")
        text = html.sub(r"", text)  # Removing html tags

        punctuations = "@#!?+&*[]-%.:/();$=><|{}^" + "'`" + "_"
        for p in punctuations:
            text = text.replace(p, "")  # Removing punctuations

        text = [word.lower() for word in text.split() if word.lower() not in sw]

        text = " ".join(text)

        return text

    def tokenize(self, texts, labels):
        input_ids = []
        attention_masks = []

        # For every text
        for text in texts:
            encoded_dict = self.tokenizer.encode_plus(
                text,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' to starts and '[SEP]' to end
                truncation=True,
                max_length=128,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors="pt",  # Return PyTorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict["input_ids"])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict["attention_mask"])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        return input_ids, attention_masks, labels

    def round_cm(self, cm: np.ndarray) -> np.ndarray:
        """
        Method to round values in confusion matrix for readability

        Args:
            cm (np.ndarray): 2D array of confusion matrix

        Returns:
            np.ndarray: 2D array of rounded confusion matrix
        """
        length = cm.shape[0]

        for i in range(length):
            for j in range(length):
                cm[i, j] = format(cm[i, j], ".2f")
        return cm

    def calculate_metrics(self, y_pred, y_true, averaging: str, device) -> None:
        """
        Method for calculating metrics based on the predictions and true labels

        Args:
            y_pred (_type_): predictions of the model
            y_true (_type_): true labels of the data
            averaging (str): averaging method for the metrics
            device (_type_): device to perform calculations on
        """
        num_classes = len(np.unique(y_true))

        accuracy_function = MulticlassAccuracy(
            num_classes=num_classes, average=averaging
        ).to(device)
        recall_function = MulticlassRecall(
            num_classes=num_classes, average=averaging
        ).to(device)
        precision_function = MulticlassPrecision(
            num_classes=num_classes, average=averaging
        ).to(device)
        fscore_function = MulticlassF1Score(
            num_classes=num_classes, average=averaging
        ).to(device)

        y_pred_class_total = torch.tensor(y_pred).to(device)
        y_true_class_total = torch.tensor(y_true).to(device)

        # Accuracy is nice, but does not tell use the whole store. So also compute precision and recall
        accuracy = accuracy_function(y_pred_class_total, y_true_class_total).item()
        recall = recall_function(y_pred_class_total, y_true_class_total).item()
        precision = precision_function(y_pred_class_total, y_true_class_total).item()
        fscore = fscore_function(y_pred_class_total, y_true_class_total).item()

        print(f"Metrics using {averaging.upper()}")
        print(f"Accuracy: {(accuracy):>0.4f}")
        print(f"Recall: {(recall):>0.4f}")
        print(f"Precision: {(precision):>0.4f}")
        print(f"F1-score: {(fscore):>0.4f}")

        cm = confusion_matrix(y_true_class_total, y_pred_class_total, normalize="true")
        cm = self.round_cm(cm)
        ConfusionMatrixDisplay(cm).plot()

    def flat_accuracy(self, predictions, labels) -> float:
        """
        Accuracy calculation for the model

        Args:
            predictions (_type_): predictions of the model
            labels (_type_): true labels of the data

        Returns:
            float: accuracy of the model
        """
        predictions_flat = np.argmax(predictions, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(predictions_flat == labels_flat) / len(labels_flat)

    def format_time(self, elapsed):
        """
        Takes a time returns a string in format hh:mm:ss

        Args:
            elapsed (float): time elapsed in seconds

        Returns:
            str: different in time
        """
        elapsed_rounded = int(round((elapsed)))  # round to seconds
        return str(dt.timedelta(seconds=elapsed_rounded))  # hh:mm:ss

    def tune(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        scheduler,
        epochs: int,
        HuggingFaceRepoName: str,
    ):
        """
        General fine-tuning method for the model

        Args:
            train_dataloader (DataLoader): DataLoader for training data
            test_dataloader (DataLoader): DataLoader for testing data
            scheduler (_type_): learning rate scheduler
            epochs (int): number of epochs to train for
            HuggingFaceRepoName (str): Repository name for HuggingFace
        """
        best_test_accuracy = 0

        # For each epoch...
        for epoch_i in range(0, epochs):
            print("")
            print("Epoch {:}/{:}".format(epoch_i + 1, epochs))

            t0 = time.time()
            avg_train_loss = self.train(train_dataloader, scheduler)
            training_time = self.format_time(time.time() - t0)
            print("")
            print("Average training loss: {0:.2f}".format(avg_train_loss))
            print("Training epoch took: {:}".format(training_time))
            print("")

            t1 = time.time()
            avg_test_accuracy = self.test(test_dataloader)
            test_time = self.format_time(time.time() - t1)

            if avg_test_accuracy > best_test_accuracy:
                torch.save(self.model, "best_bert_model")
                self.model.push_to_hub(HuggingFaceRepoName)
                self.tokenizer.push_to_hub(HuggingFaceRepoName)
                best_test_accuracy = avg_test_accuracy

        print("")
        print("Training took {:} (h:mm:ss)".format(training_time))
        print("Testing took {:} (h:mm:ss)".format(test_time))

    def train(self, train_dataloader: DataLoader, scheduler):
        """
        Training part of the model

        Args:
            train_dataloader (DataLoader): DataLoader for training data
            scheduler (_type_): learning rate scheduler

        Returns:
            avg_train_loss: average training loss
        """
        print("Training...")
        total_train_loss = 0
        self.model.train()  # put model in training mode

        for step, batch in enumerate(train_dataloader):
            # There are three PyTorch tensors in the batch:
            # [0]: input ids
            # [1]: attention masks
            # [2]: labels
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            self.optimizer.zero_grad()
            output = self.model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )  # Get the output from the model
            loss = output.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0
            )  # Help prevent exploding gradient problem by clipping the norm to 1.0
            self.optimizer.step()
            scheduler.step()  # For the learning rate scheduler

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        return avg_train_loss

    def test(self, test_dataloader):
        """
        Testing part of the model

        Args:
            test_dataloader (DataLoader): DataLoader for training data

        Returns:
            avg_test_accuracy: average test accuracy
        """
        print("Running Test...")
        self.model.eval()  # Put model in evaluation mode

        total_eval_accuracy_test = 0
        total_eval_loss_test = 0

        # Evaluate data for one epoch
        for batch in test_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            with torch.no_grad():
                output = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
            loss = output.loss
            total_eval_loss_test += loss.item()

            logits = output.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            total_eval_accuracy_test += self.flat_accuracy(logits, label_ids)
        # Print final accuracy for the test data
        avg_test_accuracy = total_eval_accuracy_test / len(test_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_test_accuracy))

        return avg_test_accuracy

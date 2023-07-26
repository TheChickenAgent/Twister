##############
# ESPECIALLY DESIGNED FOR TWO K-WAY CLASSIFIERS (CE & Focal) AND ONE MATCHING CLASSIFIER (BCE)
##############
from typing import Optional
import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from focal_loss.focal_loss import FocalLoss

from src.SentenceEmbeddings import SentenceEmbeddings
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchmetrics.classification import (
    MulticlassRecall,
    MulticlassPrecision,
    MulticlassAccuracy,
    MulticlassF1Score,
)


class CustomImplementation:
    """
    CustomImplementation: Class that implements the custom implementation of the thesis
    V3: This version is especially designed for TWO k-way classifiers (CE & Focal) and one matching classifier (BCE)
    This class is used to train the model and evaluate it on the test set.
    """

    def __init__(
        self,
        sTrans: SentenceEmbeddings,
        device: str,
        keyword_embeddings: torch.Tensor,
        label_encoder: LabelEncoder,
        class_weights: Optional[torch.Tensor],
        optimizer,
    ):
        self.sentence_transformer = sTrans
        self.device = device
        self.keyword_embeddings = keyword_embeddings
        self.le = label_encoder
        self.loss_Focal = FocalLoss(gamma=0.7, weights=class_weights)
        self.loss_CE = nn.CrossEntropyLoss(weight=class_weights)
        self.loss_BCE = nn.BCELoss(weight=class_weights)
        self.optimizer = optimizer

    def combine_sentence_class_embeddings(
        self, sentences: torch.Tensor, class_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        This method combines each sentence with one of the class semantic representations (CSR).
        The result will be in form [sentence c_i].

        Args:
            sentences (torch.Tensor): sentences in batch, already as Tensor
            class_embeddings (torch.Tensor): tensor with dimension [classes, emb_dim]

        Returns:
            torch.Tensor: tensor with sentence embedding together with class embedding, dimension [classes*batch_size, emb_dim + class_emb_size]
        """
        s_list = []

        for sentence in sentences:
            for class_embedding in class_embeddings:
                s_list.append(torch.concat((sentence, class_embedding)))

        return torch.stack(s_list)

    def reshape_matching(self, y_true: torch.Tensor) -> torch.Tensor:
        """
        This method reshapes the y_true to be in a matching from. It encodes a *1* where it matches and a *0* where it does not match.

        Args:
            y_true (torch.Tensor): tensor having the matches values from the sklearn LabelEncoder (so 0, ..., 8)

        Returns:
            torch.Tensor: tensor with ones where we have a match and 0 otherwise
        """
        matching = torch.zeros(
            size=(y_true.size(dim=0), len(self.le.classes_))
        )  # initialize to be all ZEROS
        matching[
            torch.arange(y_true.size(dim=0)), y_true
        ] = 1  # encode 1 where it matches
        matching = matching.to(self.device)  # to the same device as everything else

        return matching

    def round_cm(self, cm: np.ndarray) -> np.ndarray:
        """
        Method to round values in confusion matrix

        Args:
            cm (np.ndarray): original values in confusion matrix

        Returns:
            np.ndarray: rounded values in confusion matrix
        """
        length = cm.shape[0]

        for i in range(length):
            for j in range(length):
                cm[i, j] = format(cm[i, j], ".3f")
        return cm

    def train_loop(
        self,
        dataloader: DataLoader,
        k_way_focal: nn.Module,
        k_way_classifier: nn.Module,
        match_classifier: nn.Module,
        plot_cm: bool = False,
    ):
        """
        Training loop

        Args:
            dataloader (DataLoader): data loader to go over the data in batches
            k_way_focal (nn.Module): k-way classifier with FOCAL loss that predicts which label it is
            k_way_classifier (nn.Module): k-way classifier that predicts which label it is
            match_classifier (nn.Module): matching classifier that predicts if the text matches the keyword description
            plot_cm (bool, optional): plot confusion matrix. Defaults to False.
        """
        print("Training")

        # Put models to train mode
        k_way_focal.train()
        k_way_classifier.train()
        match_classifier.train()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        y_pred_class_total = []
        y_true_class_total = []
        train_loss = 0

        for batch, (X, y) in enumerate(dataloader):
            # Process batch
            X_embedded = self.sentence_transformer.encode(X)

            assert isinstance(X_embedded, torch.Tensor)

            # Now, for each embedded sentence, add the class embedding for each class
            X_embedded_total = self.combine_sentence_class_embeddings(
                sentences=X_embedded, class_embeddings=self.keyword_embeddings
            )

            # Compute prediction and loss
            # the K-way classifier only takes in the sentence without CSR
            y_pred_K = k_way_classifier(X_embedded)
            y_pred_focal = k_way_focal(X_embedded)
            # the matching classifier takes in the *sentence WITH CSR*
            y_pred_Matching = match_classifier(X_embedded_total)

            # Transform true values
            y_true = self.le.transform(y)
            y_true = torch.tensor(y_true, dtype=torch.int64).to(self.device)

            # Transform matching
            y_pred_Matching = torch.reshape(
                y_pred_Matching, (len(X), len(list(self.le.classes_)))
            )
            y_true_Matching = self.reshape_matching(y_true)

            # Compute loss
            loss_focal = self.loss_Focal(y_pred_focal, y_true)
            loss_ce = self.loss_CE(y_pred_K, y_true)
            loss_bce = self.loss_BCE(y_pred_Matching, y_true_Matching)
            combined_loss = loss_ce + loss_bce + loss_focal

            # Get most likely class
            y_pred_class = torch.argmax(y_pred_K, dim=1)

            # Backpropagation
            self.optimizer.zero_grad()
            # self.optimizer_2.zero_grad()
            combined_loss.backward()
            # combined_loss_2.backward()
            self.optimizer.step()
            # self.optimizer_2.step()

            if (batch + 1) % 100 == 0:
                loss, current = combined_loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            train_loss += combined_loss.item()  # add to loss

            y_pred_class_total.extend(y_pred_class.detach().cpu().numpy())
            y_true_class_total.extend(y_true.detach().cpu().numpy())

        train_loss /= num_batches

        self.calculate_metrics(
            "TRAIN", train_loss, y_pred_class_total, y_true_class_total, "macro"
        )
        print()
        self.calculate_metrics(
            "TRAIN", train_loss, y_pred_class_total, y_true_class_total, "weighted"
        )
        print("-" * 100)

        # Plot confusion matrix for the last epoch
        if plot_cm:
            cm = confusion_matrix(
                y_true_class_total, y_pred_class_total, normalize="true"
            )
            cm = self.round_cm(cm)
            ConfusionMatrixDisplay(cm).plot()

    def test_loop(
        self,
        dataloader: DataLoader,
        k_way_focal: nn.Module,
        k_way_classifier: nn.Module,
        match_classifier: nn.Module,
        plot_cm: bool = False,
    ):
        """
        Testing loop

        Args:
            dataloader (DataLoader): data loader to go over the data in batches
            k_way_focal (nn.Module): k-way classifier with FOCAL loss that predicts which label it is
            k_way_classifier (nn.Module): k-way classifier that predicts which label it is
            match_classifier (nn.Module): matching classifier that predicts if the text matches the keyword description
            plot_cm (bool, optional): plot confusion matrix. Defaults to False.
        """
        print("Testing")

        # Put models to inference mode
        k_way_focal.eval()
        k_way_classifier.eval()
        match_classifier.eval()
        num_batches = len(dataloader)
        test_loss = 0
        y_pred_class_total = []
        y_true_class_total = []

        # torch.no_grad():
        # disables computation of gradients for the backward pass. Since these calculations are unnecessary during inference
        with torch.no_grad():
            for X, y in dataloader:
                # Process batch
                X_embedded = self.sentence_transformer.encode(X)

                assert isinstance(X_embedded, torch.Tensor)

                # Now, for each embedded sentence, add the class embedding for each class
                X_embedded_total = self.combine_sentence_class_embeddings(
                    sentences=X_embedded, class_embeddings=self.keyword_embeddings
                )

                # Predict batch
                # the K-way classifier only takes in the sentence without CSR
                y_pred_K = k_way_classifier(X_embedded)
                y_pred_Focal = k_way_focal(X_embedded)
                # the matching classifier takes in the *sentence WITH CSR*
                y_pred_Matching = match_classifier(X_embedded_total)

                # Transform true values
                y_true = self.le.transform(y)
                y_true = torch.tensor(y_true, dtype=torch.int64).to(self.device)

                # Transform matching
                y_pred_Matching = torch.reshape(
                    y_pred_Matching, (len(X), len(list(self.le.classes_)))
                )
                y_true_Matching = self.reshape_matching(y_true)

                # Compute loss
                loss_ce = self.loss_CE(y_pred_K, y_true)
                loss_focal = self.loss_Focal(y_pred_Focal, y_true)
                loss_bce = self.loss_BCE(y_pred_Matching, y_true_Matching)
                combined_loss = loss_ce + loss_bce + loss_focal

                test_loss += combined_loss.item()  # add to loss

                # Get most likely class
                y_pred_class = torch.argmax(y_pred_K, dim=1)

                # Print predictions and labels
                # print("Predictions:", y_pred_class)
                # print("True:", y_true)

                y_pred_class_total.extend(y_pred_class.detach().cpu().numpy())
                y_true_class_total.extend(y_true.detach().cpu().numpy())

        test_loss /= num_batches

        self.calculate_metrics(
            "TEST", test_loss, y_pred_class_total, y_true_class_total, "macro"
        )
        print()
        self.calculate_metrics(
            "TEST", test_loss, y_pred_class_total, y_true_class_total, "weighted"
        )
        print("-" * 100)

        # Plot confusion matrix for the last epoch
        if plot_cm:
            cm = confusion_matrix(
                y_true_class_total, y_pred_class_total, normalize="true"
            )
            cm = self.round_cm(cm)
            ConfusionMatrixDisplay(cm).plot()

    def calculate_metrics(
        self, type: str, avg_loss: float, y_pred: list, y_true: list, averaging: str
    ) -> None:
        """
        Method that calculates the metrics, depending on averaging strategy

        Args:
            type (str): to put which split you are using
            avg_loss (float): average loss from train/test loop
            y_pred (list): list of predicted labels
            y_true (list): list of true labels
            averaging (str): averaging technique for metrics, i.e. macro, micro, weighted
        """
        num_classes = len(list(self.le.classes_))
        accuracy_function = MulticlassAccuracy(
            num_classes=num_classes, average=averaging
        ).to(self.device)
        recall_function = MulticlassRecall(
            num_classes=num_classes, average=averaging
        ).to(self.device)
        precision_function = MulticlassPrecision(
            num_classes=num_classes, average=averaging
        ).to(self.device)
        fscore_function = MulticlassF1Score(
            num_classes=num_classes, average=averaging
        ).to(self.device)

        y_pred_class_total = torch.tensor(y_pred).to(self.device)
        y_true_class_total = torch.tensor(y_true).to(self.device)

        # Accuracy is nice, but does not tell use the whole store. So also compute precision, recall and F1
        accuracy = accuracy_function(y_pred_class_total, y_true_class_total).item()
        recall = recall_function(y_pred_class_total, y_true_class_total).item()
        precision = precision_function(y_pred_class_total, y_true_class_total).item()
        fscore = fscore_function(y_pred_class_total, y_true_class_total).item()

        print(f"{type} metrics using {averaging.upper()}")
        print(f"Average loss: {avg_loss:>8f}")
        print(f"Accuracy: {(accuracy):>0.4f}")
        print(f"Recall: {(recall):>0.4f}")
        print(f"Precision: {(precision):>0.4f}")
        print(f"F1-score: {(fscore):>0.4f}")

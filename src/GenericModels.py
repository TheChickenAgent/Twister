from torch import nn

##Concatenating embedding per CSR, but we need two different model (one for matching, one for K-way):
##For using MiniLM embeddings:
#### K-way: 384
#### Matching: 384 + 768 = 1152

##For using MPNet embeddings:
#### K-way: 768
#### Matching: 768 + 768 = 1536

##For using MPNet embeddings:
#### K-way: 768
#### Matching: 768 + 768 = 1536


class k_classifier(nn.Module):
    """
    K-way classifier that predicts the class of a text

    Args:
        nn (torch.nn.Module): Module it inherits from
    """

    def __init__(self, num_classes: int, sTrans_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(sTrans_dim, 128)
        self.act1 = nn.Tanh()
        self.output = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.softmax(self.output(x))
        return x


class matching_classifier(nn.Module):
    """
    Matching classifier that predicts if there is a match between the text and the class embedding

    Args:
        nn (torch.nn.Module): Module it inherits from
    """

    def __init__(self, combined_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(combined_dim, 128)
        self.act1 = nn.Tanh()
        self.output = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.sigmoid(self.output(x))
        return x

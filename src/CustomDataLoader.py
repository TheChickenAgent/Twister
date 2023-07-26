import numpy as np
import pandas as pd

from datasets import load_dataset
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight


class CustomDataLoader:
    """
    CustomDataLoader: Class that loads data from HuggingFace's datasets library and can:
    - Convert it to pd.DataFrames
    - Select equal number of shots for each class
    - Compute class weights
    - Compute square root of class weights
    """

    def __init__(self, name: str, subset=None, data_files=None) -> None:
        self.name = name
        self.subset = subset
        self.data_files = data_files  # dict|None

    def to_dataframe(
        self, data_dict, subsets: list, print_dataframe: bool = False
    ) -> dict:
        """
        Forms HuggingFace's dataset dictionary to a dictionary of pandas DataFrames

        Args:
            data_dict (dict[str, data]): dataset with a set indicating if it's train, validation or test set
            subsets (list[str]): list of subsets of dataset
            print_dataframe (bool): to print the converted dataframe

        Returns:
            dict[str, pd.DataFrame]: pd.DataFrame accompanied by the set
        """
        dfs = {}

        # Go over every subset and convert it to pandas DataFrame
        for subset in tqdm(subsets):
            dfs[subset] = pd.DataFrame(data_dict[subset])

        # Different for loop for better printing
        for subset in subsets:
            print(subset, len(dfs[subset]))
            if print_dataframe:
                print(dfs[subset])

        # print(dfs[subsets[-1]].columns) #print columns of last df

        return dfs

    def load_huggingface_data(self):
        """
        Load HuggingFace data using the name defined in constructor and the optional data_files

        Returns:
            DatasetDict | Dataset | IterableDatasetDict | IterableDataset: HuggingFace dataset
        """
        if self.subset == None:
            return load_dataset(self.name, data_files=self.data_files)
        else:
            return load_dataset(self.name, self.subset, data_files=self.data_files)

    def selectEqualFewshots(
        self,
        dataframe: pd.DataFrame,
        label_name: str,
        text_name: str,
        shots: int,
        seed: int,
    ) -> pd.DataFrame:
        """
        Method that takes an equal number of shots for each class

        Args:
            dataframe (pd.DataFrame): Dataframe where the data is in
            label_name (str): string indicating column name of the label
            text_name (str): string indicating column name of the text
            shots (int): number of instances per class
            seed (int): seed to ensure reproducability

        Returns:
            pd.DataFrame: dataframe with number of shots*classes instances
        """
        classes = list(np.unique(dataframe[label_name]))
        fewshots_texts = []
        fewshots_labels = []

        for classe in classes:
            df_class = dataframe[dataframe[label_name] == classe]
            df_class = df_class.sample(n=shots, random_state=seed)
            fewshots_texts.extend(df_class[text_name].to_list())
            fewshots_labels.extend(df_class[label_name].to_list())

        df_fewshots = pd.DataFrame(
            data={text_name: fewshots_texts, label_name: fewshots_labels}
        )

        return df_fewshots

    def compute_class_weight_sqrt(
        self, dataframe: pd.DataFrame, label: str
    ) -> np.ndarray:
        """
        Method uses sklearn's implementation to calculate balanced class weight and then takes the square root of it

        Args:
            dataframe (pd.DataFrame): Dataframe where the data is in
            label (str): string indicating column name of the label

        Returns:
            np.ndarray: array with distribution of classes
        """
        class_weights = self.compute_class_weight_normal(
            dataframe=dataframe, label=label
        )
        class_weights = np.sqrt(class_weights)

        return class_weights

    def compute_class_weight_normal(
        self, dataframe: pd.DataFrame, label: str
    ) -> np.ndarray:
        """
        Method uses sklearn's implementation to calculate balanced class weight

        Args:
            dataframe (pd.DataFrame): Dataframe where the data is in
            label (str): string indicating column name of the label

        Returns:
            np.ndarray: array with distribution of classes
        """
        data_labels = dataframe[label]
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(data_labels),
            y=data_labels.values,
        )

        return class_weights

from torch.utils.data import Dataset, DataLoader
import torch


class ComputationGraphDataset(Dataset):
    def __init__(self, list_of_graphs):
        """

        :param list_of_graphs: The list of computation graphs to use, loaded from the pickle file
        """
        self.graphs = list_of_graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        sample = {'graph': graph, 'operations': graph.get_operations()}
        return sample


class ExpressionDataset(Dataset):
    def __init__(self, list_of_expressions_and_labels):
        self.data_points = list_of_expressions_and_labels

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        return self.data_points[idx]

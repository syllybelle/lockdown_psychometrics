from typing import List, Dict, Union
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.multiarray import ndarray
from pandas import DataFrame
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


class Clusterer:
    def __init__(self, data_as_array: ndarray, eps: float, min_samples: int) -> None:
        self.data: ndarray = data_as_array
        self.eps: float = round(eps, 4)
        self.min_samples: int = min_samples
        self.raw_db_scan_output: DBSCAN = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(self.data)
        self.labels: ndarray = self.raw_db_scan_output.labels_
        self.n_clusters: int = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        self.cluster_sizes: Dict[int, int] = dict(zip(*np.unique(self.labels, return_counts=True)))
        self.n_noise: int = list(self.labels).count(-1)
        self.silhouette_coefficient: float = round(metrics.silhouette_score(self.data, self.labels), 6)

    def plot_db_scan_output(self) -> None:
        unique_labels = set(self.labels)
        colors = []
        for each in np.linspace(0, 1, len(unique_labels)):
            colors.append(plt.cm.Spectral(each))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            core_samples_mask = np.zeros_like(self.labels, dtype=bool)
            core_samples_mask[self.raw_db_scan_output.core_sample_indices_] = True

            class_member_mask = (self.labels == k)
            xy = self.data[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col),
                     markeredgecolor="k", markersize=14)
            plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col),
                     markeredgecolor="k", markersize=6)
        plt.title(f'''clusters: {self.n_clusters}, noise: {self.n_noise}, eps: {self.eps}, 
min samples: {self.min_samples}, Silhouette Coefficient: {self.silhouette_coefficient}''')
        plt.show()

    def output_df(self, variable_list, per_cluster: bool = False) -> Union[DataFrame, Dict[int, DataFrame]]:
        """ reshape to a 2d array and concatenate, sort by cluster and add column names """
        combined_array: ndarray = np.concatenate([self.data, self.labels.reshape(-1, 1)], axis=1)
        cluster_data_df: DataFrame = DataFrame(combined_array)
        column_names: List[str] = list(variable_list)
        column_names.append('cluster')
        cluster_data_df.set_axis(column_names, axis='columns', inplace=True)
        if per_cluster is False:
            return cluster_data_df
        else:
            output_by_cluster: Dict[int, DataFrame] = {}
            for cluster_no in self.cluster_sizes.keys():
                new_df: DataFrame = cluster_data_df[cluster_data_df.cluster == cluster_no]
                output_by_cluster[cluster_no] = new_df
            return output_by_cluster


class EpsOptimiser:

    def __init__(self,
                 eps_range: tuple,
                 data_as_array: ndarray,
                 min_samples_per_cluster: int,
                 increment: float = 0.01) -> None:
        self.eps_range: tuple = eps_range  # to be read off graph
        self.increment: float = increment

        self.all_solutions_list: List[List[any]] = []
        self.viable_solutions_list: List[List[any]] = []
        column_names: List[str] = ['eps', 'clusters', 'noise', 'silhouette']

        eps: float = eps_range[0]
        while eps <= eps_range[1]:
            solution: Clusterer = Clusterer(data_as_array, eps=eps, min_samples=min_samples_per_cluster)
            current_eps: float = solution.eps
            cluster_n: int = solution.n_clusters
            noise_n: int = solution.n_noise
            silhouette: float = solution.silhouette_coefficient
            row: List[any] = [current_eps, cluster_n, noise_n, silhouette]
            # silhouette coefficient of < 0 means that the average density outside clusters is greater than the
            # average density within clusters.
            if cluster_n != 1 and silhouette >= 0.0:
                self.viable_solutions_list.append(row)
                self.all_solutions_list.append(row)
            else:
                self.all_solutions_list.append(row)
            eps += self.increment

        viable_solutions_df: DataFrame = DataFrame(self.viable_solutions_list, columns=column_names)
        self.viable_solutions_sorted: DataFrame = viable_solutions_df.sort_values(by=['silhouette'], ascending=[False])
        all_solutions_df: DataFrame = DataFrame(self.all_solutions_list, columns=column_names)
        self.all_solutions_sorted: DataFrame = all_solutions_df.sort_values(by=['silhouette'], ascending=[False])


def find_eps_inflection(data: ndarray) -> None:
    x: ndarray = data
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(x)
    distances, indices = nbrs.kneighbors(x)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)

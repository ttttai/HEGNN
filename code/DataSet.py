import lightning as L
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Subset
from torch_geometric.data import HeteroData
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from torch_geometric.utils import subgraph
from typing import List, Literal, Optional, Tuple, Union, overload
from torch import Tensor
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils.map import map_index
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch_geometric.transforms as T


def subgraph_for_hetero(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    *,
    return_edge_mask: bool = False,
) -> Union[Tuple[Tensor, OptTensor], Tuple[Tensor, OptTensor, Tensor]]:
    device = edge_index.device

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    node_mask = index_to_mask(subset, size=num_nodes)
    node_mask = node_mask.to(device)

    edge_mask = node_mask[edge_index[0]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        edge_index, _ = map_index(
            edge_index.view(-1),
            subset,
            max_index=num_nodes,
            inclusive=True,
        )
        edge_index = edge_index.view(2, -1)

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr


class DataSet(torch.utils.data.Dataset):
    """
    データセット
    """

    def __init__(
        self, fold=10, input_dir="../processed_data/", device="cuda:1", seed=0
    ):
        super().__init__()
        self.device = device
        self.seed = seed
        self.post_data = pd.read_csv(input_dir + "post_data.csv")
        self.station_data = pd.read_csv(input_dir + "station_data.csv")
        self.edge_data_between_post_and_station = pd.read_csv(
            input_dir + "edge_data_between_post_and_station.csv"
        )
        self.edge_data_between_station_and_station = pd.read_csv(
            input_dir + "edge_data_between_station_and_station.csv"
        )
        self.edge_indices_between_post_and_station = pd.read_csv(
            input_dir + "edge_indices_between_post_and_station.csv"
        )
        self.edge_indices_between_station_and_station = pd.read_csv(
            input_dir + "edge_indices_between_station_and_station.csv"
        )
        self.commuter_pass_valid_days = pd.read_csv(
            input_dir + "commuter_pass_valid_days_2022.csv"
        )
        self.post_coors_data = pd.read_csv(input_dir + "post_coors_data.csv")
        self.station_coors_data = pd.read_csv(input_dir + "station_coors_data.csv")
        self.post_index = self.post_data[["post_index", "postal_lat", "postal_lon"]]

        self._remove_unused_data()
        self._scale()
        self.data = self._create_hetero_data()
        self.splitted_data = self._split(fold)

    def __len__(self):
        return len(self.splitted_data)

    def __getitem__(self, idx):
        return self.splitted_data[idx]

    def _remove_unused_data(self):
        """
        必要ないカラムを削除
        """
        self.post_data = self.post_data.drop(columns=["post_index", "postal_code"])
        self.station_data = self.station_data.drop(
            columns=["station_index", "station_name"]
        )
        self.edge_data_between_post_and_station = (
            self.edge_data_between_post_and_station.drop(
                columns=["post_index", "station_index"]
            )
        )

        self.post_data = self.post_data.drop(columns=["postal_lat", "postal_lon"])
        self.station_data = self.station_data.drop(
            columns=["station_lat", "station_lon"]
        )

    def _scale(self):
        """
        データをスケーリングする
        """
        rs1 = RobustScaler()
        self.post_data = rs1.fit_transform(self.post_data)
        rs2 = RobustScaler()
        self.station_data = rs2.fit_transform(self.station_data)
        rs3 = RobustScaler()
        self.edge_data_between_post_and_station = rs3.fit_transform(
            self.edge_data_between_post_and_station
        )
        rs4 = RobustScaler()
        self.edge_data_between_station_and_station = rs4.fit_transform(
            self.edge_data_between_station_and_station
        )

    def _create_hetero_data(self):
        """
        HeteroDataを作成

        Returns:
            HeteroData: ヘテロデータ
        """
        self.post_data = torch.tensor(self.post_data).type(torch.float).to(self.device)
        self.station_data = (
            torch.tensor(self.station_data).type(torch.float).to(self.device)
        )
        self.post_coors_data = torch.tensor(self.post_coors_data.values).to(self.device)
        self.station_coors_data = torch.tensor(self.station_coors_data.values).to(
            self.device
        )
        self.edge_indices_between_post_and_station = torch.tensor(
            self.edge_indices_between_post_and_station.T.values
        ).to(self.device)
        self.edge_indices_between_station_and_station = torch.tensor(
            self.edge_indices_between_station_and_station.T.values
        ).to(self.device)
        self.edge_data_between_post_and_station = torch.tensor(
            self.edge_data_between_post_and_station
        ).to(self.device)

        data = HeteroData()
        data["post"].x = self.post_data
        data["station"].x = self.station_data
        data["post"].coors = self.post_coors_data
        data["station"].coors = self.station_coors_data
        data["post", "to", "station"].edge_index = (
            self.edge_indices_between_post_and_station
        )
        data["station", "to", "station"].edge_index = (
            self.edge_indices_between_station_and_station
        )
        data["post", "to", "station"].edge_attr = (
            self.edge_data_between_post_and_station
        )
        data = T.ToUndirected()(data)
        return data

    def _split(self, fold):
        """
        データをkmeansで分割する

        Args:
            fold (int): fold数

        Returns:
            list: 分割されたデータ
        """
        kmeans = KMeans(n_clusters=fold, random_state=self.seed).fit(
            self.post_index[["postal_lat", "postal_lon"]].values
        )
        post_index_splitted = tuple(
            torch.tensor(self.post_index[kmeans.labels_ == label]["post_index"].values)
            for label in np.unique(kmeans.labels_)
        )

        self.commuter_pass_valid_days = torch.tensor(
            self.commuter_pass_valid_days.values
        ).to(self.device)

        splitted_data = []
        for i in range(fold):
            train_data = []
            test_data = []
            for j in range(fold):
                # 訓練データの郵便番号に接続されている駅とその駅に接続されている駅のグラフを取得
                (
                    sub_edge_index_between_post_and_station,
                    sub_edge_attr_between_post_and_station,
                    edge_mask,
                ) = subgraph_for_hetero(
                    post_index_splitted[j].unsqueeze(0),
                    self.data.edge_index_dict["post", "to", "station"],
                    edge_attr=self.data.edge_attr_dict["post", "to", "station"],
                    num_nodes=self.post_data.size(dim=0),
                    return_edge_mask=True,
                )
                sub_edge_index_between_station_rev_to_post = (
                    sub_edge_index_between_post_and_station[[1, 0], :]
                )
                sub_edge_attr_between_station_rev_to_post = (
                    sub_edge_attr_between_post_and_station
                )
                sub_station_index = sub_edge_index_between_post_and_station[1].unique()
                # sub_edge_index_between_station_and_station, sub_edge_attr_between_station_and_station = subgraph(sub_station_index, data.edge_index_dict['station', 'to', 'station'], edge_attr=data.edge_attr_dict['station', 'to', 'station'], num_nodes=station_data.size(dim=0))
                sub_edge_index_between_station_and_station = subgraph(
                    sub_station_index,
                    self.data.edge_index_dict["station", "to", "station"],
                    num_nodes=self.station_data.size(dim=0),
                )[0]
                sub_edge_index_dict = {
                    ("post", "to", "station"): sub_edge_index_between_post_and_station,
                    (
                        "station",
                        "rev_to",
                        "post",
                    ): sub_edge_index_between_station_rev_to_post,
                    (
                        "station",
                        "to",
                        "station",
                    ): sub_edge_index_between_station_and_station,
                }
                sub_edge_attr_dict = {
                    ("post", "to", "station"): sub_edge_attr_between_post_and_station,
                    (
                        "station",
                        "rev_to",
                        "post",
                    ): sub_edge_attr_between_station_rev_to_post,
                    # ('station', 'to', 'station'): sub_edge_attr_between_station_and_station
                }
                commuter_pass_valid_days_partition = self.commuter_pass_valid_days[
                    edge_mask
                ]

                out_dict = {
                    "x_dict": self.data.x_dict,
                    "sub_edge_index_dict": sub_edge_index_dict,
                    "sub_edge_attr_dict": sub_edge_attr_dict,
                    "coors_dict": self.data.coors_dict,
                    "commuter_pass_valid_days": commuter_pass_valid_days_partition,
                }

                if j == i:
                    test_data.append(out_dict)
                else:
                    train_data.append(out_dict)

            splitted_data.append([train_data, test_data])

        return splitted_data

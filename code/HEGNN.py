import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch.types import Union


class HEGNN(MessagePassing):
    def __init__(
        self,
        edge_dim: dict[str, int],
        embedding_dim: int,
        edge_hidden_dim: int,
        node_hidden_dim: int,
        out_features_dim: int,
        m_hidden_dim: int,
        edge_types: dict[tuple[str, str, str], str],
        m_dim: int,
        C: float,
        aggr: str = "sum",
    ) -> None:
        """_summary_

        Args:
            node_dim (dict[str, int]): _description_
            edge_dim (dict[str, int]): _description_
            embedding_dim (int): _description_
            edge_hidden_dim (int): _description_
            node_hidden_dim (int): _description_
            out_features_dim (int): _description_
            m_hidden_dim (int): _description_
            node_types (dict[tuple[str, str, str], str]): _description_
            edge_types (dict[tuple[str, str, str], str]): _description_
            m_dim (int): _description_
            aggr (str, optional): _description_. Defaults to 'sum'.
        """
        super().__init__(aggr=aggr)

        # 各エッジごとのmlp
        # module nameはstringでなければならないので以下のようにアンダースコア区切りに変換する
        # ('type1', 'relation', 'type2') => 'type1_relation_type2'
        self.edge_mlp_dict = nn.ModuleDict(
            {
                "_".join(edge_type): nn.Sequential(
                    nn.Linear(
                        2 * embedding_dim + edge_dim[edge_type] + 1, edge_hidden_dim
                    ),
                    nn.Dropout(0.1),
                    nn.SiLU(),
                    nn.Linear(edge_hidden_dim, m_dim),
                )
                for edge_type in edge_types
            }
        )

        # ノードのmlp
        self.node_mlp = nn.Sequential(
            nn.Linear(embedding_dim + m_dim, node_hidden_dim),
            nn.Dropout(0.1),
            nn.SiLU(),
            nn.Linear(node_hidden_dim, out_features_dim),
        )

        # 座標のmlp
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_hidden_dim),
            nn.Dropout(0.1),
            nn.SiLU(),
            nn.Linear(m_hidden_dim, 1),
        )

        # 座標更新の際の重み
        self.C = C

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
        edge_attr_dict: dict,
        coors_dict: dict,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """_summary_

        Args:
            x_dict (dict): _description_
            edge_index_dict (dict): _description_
            edge_attr_dict (dict): _description_
            coors_dict (dict): _description_

        Returns:
            tuple[dict[str, Tensor], dict[str, Tensor]]: _description_
        """
        m_dict = {}
        coors_delta_dict = {}
        for edge_type in edge_index_dict:
            src, rel, dst = edge_type
            if src == dst:
                x = x_dict[dst]
                coors = coors_dict[dst]
            else:
                x = (x_dict[src], x_dict[dst])
                coors = (coors_dict[src], coors_dict[dst])

            # mについての集約
            if edge_type in edge_attr_dict.keys():
                m_aggregated = self.propagate(
                    edge_index=edge_index_dict[edge_type],
                    x=x,
                    coors=coors,
                    edge_attr=edge_attr_dict[edge_type],
                    edge_type="_".join(edge_type),
                    type="m",
                )
            else:
                m_aggregated = self.propagate(
                    edge_index=edge_index_dict[edge_type],
                    x=x,
                    coors=coors,
                    edge_attr=None,
                    edge_type="_".join(edge_type),
                    type="m",
                )

            if dst not in m_dict:
                m_dict[dst] = m_aggregated
            else:
                m_dict[dst] += m_aggregated

            # 座標についての集約
            if edge_type in edge_attr_dict.keys():
                coors_aggregated = self.propagate(
                    edge_index=edge_index_dict[edge_type],
                    x=x,
                    coors=coors,
                    edge_attr=edge_attr_dict[edge_type],
                    edge_type="_".join(edge_type),
                    type="coors",
                )
            else:
                coors_aggregated = self.propagate(
                    edge_index=edge_index_dict[edge_type],
                    x=x,
                    coors=coors,
                    edge_attr=None,
                    edge_type="_".join(edge_type),
                    type="coors",
                )

            if dst not in coors_delta_dict:
                coors_delta_dict[dst] = coors_aggregated
            else:
                coors_delta_dict[dst] += coors_aggregated

        # ノードの更新
        out_x_dict = {}
        for node_type, value in x_dict.items():
            node_mlp_input = torch.cat((value, m_dict[node_type]), dim=1)
            out_x_dict[node_type] = self.node_mlp(node_mlp_input)

        # 座標の更新
        out_coors_dict = {}
        for node_type, value in coors_dict.items():
            out_coors_dict[node_type] = self.C * coors_delta_dict[node_type]

        return out_x_dict, out_coors_dict

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        coors_i: Tensor,
        coors_j: Tensor,
        edge_attr: Tensor,
        edge_type: str,
        type: str,
    ) -> Tensor:
        """メッセージ

        Args:
            x_i (Tensor): 送信先の埋め込み表現
            x_j (Tensor): 送信元の埋め込み表現
            coors_i (Tensor): 送信先の座標
            coors_j (Tensor): 送信元の座標
            edge_attr (Tensor): エッジの特徴量
            edge_type (str): エッジの種類
            type (str): m or coors

        Returns:
            Tensor: 伝達する値
        """
        distance_squared = torch.sum((coors_i - coors_j) ** 2, dim=1, keepdim=True)
        if edge_attr is None:
            edge_mlp_input = torch.cat((x_i, x_j, distance_squared), dim=1)
        else:
            edge_mlp_input = torch.cat((x_i, x_j, distance_squared, edge_attr), dim=1)

        edge_mlp_output = self.edge_mlp_dict[edge_type](edge_mlp_input.to(torch.float))
        if type == "m":
            return edge_mlp_output
        elif type == "coors":
            coors_mlp_output = (coors_i - coors_j) * self.coors_mlp(edge_mlp_output)
            return coors_mlp_output
        else:
            raise ValueError(f"Invalid input: {type}")

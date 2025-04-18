import torch
import lightning as L
from DataSet import DataSet
from HEGNN import HEGNN
from torch.nn import Linear, ModuleDict
from torch.utils.data import DataLoader, Dataset
import mlflow


class Model(L.LightningModule):
    area_num = 0
    test_loss = []

    def __init__(
        self,
        node_dim: dict[str, int],
        edge_dim: dict[str, int],
        embedding_dim: int,
        edge_hidden_dim: int,
        node_hidden_dim: int,
        out_features_dim: int,
        m_hidden_dim: int,
        m_dim: int,
        node_types: dict[tuple[str, str, str], str],
        edge_types: dict[tuple[str, str, str], str],
        C: float,
        lr=0.001,
        lin_hidden_dim=30,
        lin_hidden_dim2=30,
        aggr="mean",
    ):
        super().__init__()

        self.lr = lr
        # ノードのタイプごとの線形変換
        self.lin = ModuleDict(
            {
                node_type: Linear(node_dim[node_type], embedding_dim)
                for node_type in node_types
            }
        )

        # 予測する際の線形変換
        self.lin1 = Linear(out_features_dim * 2 + 7, lin_hidden_dim)
        self.lin2 = Linear(lin_hidden_dim, lin_hidden_dim2)
        self.lin3 = Linear(lin_hidden_dim2, 1)

        self.HEGNN1 = HEGNN(
            edge_dim=edge_dim,
            embedding_dim=embedding_dim,
            edge_hidden_dim=edge_hidden_dim,
            node_hidden_dim=node_hidden_dim,
            out_features_dim=out_features_dim,
            m_hidden_dim=m_hidden_dim,
            m_dim=m_dim,
            edge_types=edge_types,
            C=C,
            aggr=aggr,
        )

        # embedding_dimは一層前の出力の次元
        self.HEGNN2 = HEGNN(
            edge_dim=edge_dim,
            embedding_dim=out_features_dim,
            edge_hidden_dim=edge_hidden_dim,
            node_hidden_dim=node_hidden_dim,
            out_features_dim=out_features_dim,
            m_hidden_dim=m_hidden_dim,
            m_dim=m_dim,
            edge_types=edge_types,
            C=C,
            aggr=aggr,
        )

        self.HEGNN3 = HEGNN(
            edge_dim=edge_dim,
            embedding_dim=out_features_dim,
            edge_hidden_dim=edge_hidden_dim,
            node_hidden_dim=node_hidden_dim,
            out_features_dim=out_features_dim,
            m_hidden_dim=m_hidden_dim,
            m_dim=m_dim,
            edge_types=edge_types,
            C=C,
            aggr=aggr,
        )

        self.criterion = torch.nn.MSELoss()
        self.train_outputs = []
        self.test_outputs = []

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, coors_dict):
        x_dict = {key: self.lin[key](x) for key, x in x_dict.items()}
        x_dict, coors_dict = self.HEGNN1(
            x_dict, edge_index_dict, edge_attr_dict, coors_dict
        )
        x_dict, coors_dict = self.HEGNN2(
            x_dict, edge_index_dict, edge_attr_dict, coors_dict
        )
        x_dict, coors_dict = self.HEGNN3(
            x_dict, edge_index_dict, edge_attr_dict, coors_dict
        )

        pred_edge_index = edge_index_dict["post", "to", "station"]
        pred_edge_attr = edge_attr_dict["post", "to", "station"]
        x_post = x_dict["post"][pred_edge_index[0]]
        x_station = x_dict["station"][pred_edge_index[1]]
        concat = torch.cat((x_post, x_station, pred_edge_attr.to(torch.float)), dim=1)
        x = self.lin1(concat)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)
        return x

    def training_step(self, data):
        pred = self.forward(
            data["x_dict"],
            data["sub_edge_index_dict"],
            data["sub_edge_attr_dict"],
            data["coors_dict"],
        )
        loss = self.criterion(pred.float(), data["commuter_pass_valid_days"].float())
        self.train_outputs.append(loss.item())
        return loss

    def test_step(self, data):
        pred = self.forward(
            data["x_dict"],
            data["sub_edge_index_dict"],
            data["sub_edge_attr_dict"],
            data["coors_dict"],
        )
        loss = self.criterion(pred.float(), data["commuter_pass_valid_days"].float())
        print(f"loss_{Model.area_num}: {loss.item()}")
        mlflow.log_metric(f"test_loss_{Model.area_num}", loss.item())
        Model.area_num += 1
        Model.test_loss.append(loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5e-4)


def main():
    # mlflowのURLを設定
    mlflow_url = ""
    mlflow.set_tracking_uri(uri=mlflow_url)
    experiment_name = "tyamamoto_example2"
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
        )
    mlflow.set_experiment(experiment_name)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    fold = 10
    data_set = DataSet(fold=fold, device=device)
    with mlflow.start_run():
        for fold, (train_data, test_data) in enumerate(data_set):
            print(f"Fold: {fold + 1}")
            node_dim = {"post": 17, "station": 3}
            edge_dim = {
                ("post", "to", "station"): 7,
                ("station", "rev_to", "post"): 7,
                ("station", "to", "station"): 0,
            }
            node_types = ["post", "station"]
            edge_types = [
                ("post", "to", "station"),
                ("station", "rev_to", "post"),
                ("station", "to", "station"),
            ]
            C = 1 / 8000
            model = Model(
                node_dim=node_dim,
                edge_dim=edge_dim,
                embedding_dim=4,
                edge_hidden_dim=4,
                node_hidden_dim=8,
                out_features_dim=8,
                m_hidden_dim=8,
                m_dim=8,
                node_types=node_types,
                edge_types=edge_types,
                C=C,
                lin_hidden_dim=30,
                lin_hidden_dim2=30,
            )
            model.to(device)

            train_dataloader = DataLoader(train_data, batch_size=None)
            test_dataloader = DataLoader(test_data, batch_size=None)

            trainer = L.Trainer(devices=1, max_epochs=200)
            trainer.fit(model, train_dataloader)
            trainer.test(model, test_dataloader)

        ave_loss = sum(Model.test_loss) / len(Model.test_loss)
        mlflow.log_metric("ave_loss", ave_loss)


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Dict, Union

import pandas as pd
import plotly.express as px
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch_geometric.data import Batch, Data

from .decoders.binary_classifier import BinaryClassifier
from .experiment_config import ExperimentConfig
from .layers.gnn_encoders import GraphEncoder
from .model_config import OptimiserType


class Regressor(pl.LightningModule):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config

        self.protein_encoder = GraphEncoder(
            config.model.graph_encoder, input_dim=75
        )
        embedding_dims = config.model.graph_encoder.layer_dims[-1]

        self.decoder = BinaryClassifier(
            config.model.classifier, input_dim=embedding_dims
        )

        # Initialise Metrics
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()
        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.test_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.train_pearson = torchmetrics.PearsonCorrcoef()
        self.val_pearson = torchmetrics.PearsonCorrcoef()
        self.test_pearson = torchmetrics.PearsonCorrcoef()
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        self.test_r2 = torchmetrics.R2Score()

    def forward(self, protein: Data) -> torch.Tensor:
        """Performs a single forward pass through the model.

        General workflow is: embed protein, aggregate node embeddings
        (concatenate), decode.

        :param protein: Protein data
        :type protein: Data
        :return: Tensor of Predictions
        """
        # Embed inputs
        protein_emb = self.protein_encoder(protein)

        return self.decoder(protein_emb)

    def training_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        protein = Batch.from_data_list(
            batch[0][0], follow_batch=["ligand_coords", "target_coords"]
        )
        label = torch.Tensor(batch[2]).unsqueeze(1)

        if torch.cuda.is_available():
            label = label.cuda()

        pred = self.forward(protein)

        loss = F.mse_loss(pred, label)

        self.log("train_loss", loss, on_epoch=True)
        self.log(
            "train_mse",
            self.train_mse(pred, label),
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_rmse",
            self.train_rmse(pred, label),
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train_mae", self.train_mae(pred, label), on_epoch=True)
        self.log(
            "train_pearson",
            self.train_pearson(pred, label),
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_r2",
            self.train_r2(pred, label),
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "pred": pred.detach(), "label": label.detach()}

    def validation_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        protein = Batch.from_data_list(
            batch[0][0], follow_batch=["ligand_coords", "target_coords"]
        )
        label = torch.Tensor(batch[2]).unsqueeze(1)

        if torch.cuda.is_available():
            label = label.cuda()

        pred = self.forward(protein)
        loss = F.mse_loss(pred, label)

        self.log("val_loss", loss, on_epoch=True)
        self.log(
            "val_mse", self.val_mse(pred, label), on_epoch=True, prog_bar=True
        )
        self.log(
            "val_rmse",
            self.val_rmse(pred, label),
            on_epoch=True,
            prog_bar=True,
        )
        self.log("val_mae", self.val_mae(pred, label), on_epoch=True)
        self.log(
            "val_pearson",
            self.val_pearson(pred, label),
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_r2", self.val_r2(pred, label), on_epoch=True, prog_bar=True
        )
        return {"pred": pred.detach(), "label": label.detach()}

    def test_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        protein = Batch.from_data_list(
            batch[0][0], follow_batch=["ligand_coords", "target_coords"]
        )
        label = torch.Tensor(batch[2]).unsqueeze(1)

        if torch.cuda.is_available():
            label = label.cuda()

        pred = self.forward(protein)
        loss = F.mse_loss(pred, label)

        self.log("test_loss", loss, on_epoch=True)
        self.log(
            "test_mse",
            self.test_mse(pred, label),
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_rmse",
            self.test_rmse(pred, label),
            on_epoch=True,
            prog_bar=True,
        )
        self.log("test_mae", self.test_mae(pred, label), on_epoch=True)
        self.log(
            "test_pearson",
            self.test_pearson(pred, label),
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_r2", self.test_r2(pred, label), on_epoch=True, prog_bar=True
        )
        return {"pred": pred.detach(), "label": label.detach()}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """Plots training predictions vs targets regression plot at the end of each training epoch and logs the image."""
        self._log_regression_plot(outputs, mode="train")

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """Plots validation predictions vs targets regression plot at the end of each validation epoch and logs the image."""
        self._log_regression_plot(outputs, mode="val")

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """Plots validation predictions vs targets regression plot at the end of each test epoch and logs the image."""
        self._log_regression_plot(outputs, mode="test")

    def _log_regression_plot(self, outputs: EPOCH_OUTPUT, mode: str) -> None:
        """Plots predictions vs targets regression plot at the end of each epoch and logs the image."""
        pred = [step["pred"] for step in outputs]
        label = [step["label"] for step in outputs]

        pred = torch.cat(pred).detach().cpu().squeeze(1).numpy().tolist()
        label = torch.cat(label).detach().cpu().squeeze(1).numpy().tolist()
        df = pd.DataFrame(zip(pred, label))
        df.columns = ["pred", "label"]
        fig = px.scatter(
            df,
            x="label",
            y="pred",
            opacity=1,
            trendline="ols",
        )
        if mode == "train":
            title = "Train Predictions"
        elif mode == "val":
            title = "Val Predictions"
        elif mode == "test":
            title = "Test Predictions"
        else:
            raise ValueError("Unsupported Mode. Please use train, val or test")

        self.logger.experiment.log({title: fig})

    def configure_optimizers(self) -> Union[torch.optim.SGD, torch.optim.Adam]:
        lr_scale = 1.0
        # Build ADAM
        if OptimiserType.ADAM == self.config.model.optimiser.type:
            # log.info("Using Adam Optimizer")
            return torch.optim.Adam(
                self.parameters(),
                lr=self.config.model.optimiser.learning_rate * lr_scale,
                betas=(
                    self.config.model.optimiser.beta1,
                    self.config.model.optimiser.beta2,
                ),
                eps=self.config.model.optimiser.epsilon,
                weight_decay=self.config.model.optimiser.weight_decay,
                amsgrad=self.config.model.optimiser.amsgrad,
            )
        # Build SGD
        elif OptimiserType.SGD == self.config.model.optimiser.type:
            # log.info("Using SGD Optimizer")
            return torch.optim.SGD(
                self.parameters(),
                lr=self.config.model.optimiser.learning_rate * lr_scale,
            )
        # Raise Error
        else:
            message = f"Optimizer {self.config.model.optimiser.type} is not supported."
            # log.error(message)
            raise ValueError(message)

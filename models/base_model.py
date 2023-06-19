import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched

from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import math

import DCPoint.data.data_utils as d_utils
from DCPoint.data.ModelNet40Loader import ModelNet40ClsContrast
from DCPoint.data.ShapeNetLoader import PartNormalDatasetContrast, WholeNormalDatasetContrast
from DCPoint.data.ScanNetLoader import ScannetWholeSceneContrast, ScannetWholeSceneContrastHeight, ScanNetFrameContrast
# from BYOL.data.ScanObjectNN_Loader import ScanObjectNNSVM, ScanObjectNNCL
from DCPoint.models.loss_utils import NTXentLoss, NTXentLossWithP2, NTXentLossWithP

from DCPoint.models.lars_scheduling import LARSWrapper

# from BYOL.models.networks import TargetNetwork, OnlineNetwork
from DCPoint.models.networks_dgcnn import TargetNetwork_DGCNN, OnlineNetwork_DGCNN

from DCPoint.models.network_curvetnet import TargetNetwork_CurveNet, OnlineNetwork_CurveNet


class P2pBasicalModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        assert self.hparams["network"] in ["DGCNN", "PointNet", "DGCNN-Semseg", "DGCNN-Partseg", "votenet"]
        if self.hparams["network"] == "DGCNN":
            print("Network: DGCNN\n\n\n")
            self.target_network = TargetNetwork_DGCNN(hparams)
            self.online_network = OnlineNetwork_DGCNN(hparams)
        elif self.hparams["network"] == "CurveNet":
            print("Network: CurveNet\n\n\n")
            self.target_network = TargetNetwork_CurveNet(hparams)
            self.online_network = OnlineNetwork_CurveNet(hparams)
        

        self.update_module(self.target_network, self.online_network, decay_rate=0)
        self.tau = self.hparams["decay_rate"]

    def update_module(self, target_module, online_module, decay_rate):
        online_dict = online_module.state_dict()
        target_dict = target_module.state_dict()
        for key in target_dict:
            target_dict[key] = decay_rate * target_dict[key] + (1 - decay_rate) * online_dict[key]
        target_module.load_state_dict(target_dict)

    def forward(self, pointcloud1, pointcloud2):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        y1_online, f1_online, z1_online, qz1_online = self.online_network(pointcloud1)
        y2_online, f2_online, z2_online, qz2_online = self.online_network(pointcloud2)

        with torch.no_grad():
            y1_target, f1_target, z1_target = self.target_network(pointcloud1)
            y2_target, f2_target, z2_target = self.target_network(pointcloud2)

        return {
            'y1_online': y1_online,
            'f1_online': f1_online,
            'qz1_online': qz1_online,
            'z1_online': z1_online,
            'z2_target': z2_target,
            'y2_target': y2_target,

            'y2_online': y2_online,
            'f2_online': f2_online,
            'qz2_online': qz2_online,
            'z2_online': z2_online,
            'z1_target': z1_target,
            'y1_target': y1_target
        }

    def regression_loss(self, x, y):
        norm_x = F.normalize(x, dim=1)
        norm_y = F.normalize(y, dim=1)
        loss = 2 - 2 * (norm_x * norm_y).sum() / x.size(0)
        return loss

    def CL_loss(self,x,y):
        loss_F = NTXentLoss(temperature=self.hparams["temperature"])
        loss = loss_F(x,y)
        return loss

    def CL_N_loss(self, F):
        loss_F = NTXentLoss(temperature=self.hparams["temperature"])
        _, C, _ = F.shape
        loss = 0
        for i in range(1,C):
            loss += loss_F(F[:, 0, :].squeeze(), F[:, i, :].squeeze())
        return loss/C

    def CL_N_P_loss(self,F):
        loss_F = NTXentLossWithP(temperature=self.hparams["temperature"], parameter_neg=self.hparams["para_neg"])
        _, C, _ = F.shape
        loss = 0
        for i in range(1, C):
            loss += loss_F(F[:, 0, :].squeeze(), F[:, i, :].squeeze())
        return loss / (C-1)

    def get_current_decay_rate(self, base_tau):
        tau = 1 - (1 - base_tau) * (math.cos(math.pi * self.global_step / (self.epoch_steps * self.hparams["epochs"])) + 1) / 2
        return tau

    def training_step_end(self, batch_parts_outputs):
        # Add callback for user automatically since it's key to BYOL weight update
        self.tau = self.get_current_decay_rate(self.hparams["decay_rate"])
        self.update_module(self.target_network, self.online_network, decay_rate=self.tau)
        return batch_parts_outputs

    def training_step(self, batch, batch_idx):
        pc_aug1, pc_aug2 = batch

        if self.hparams["network"] in {"DGCNN", "CurveNet"}:
            pc_aug1 = pc_aug1.permute(0, 2, 1)
            pc_aug2 = pc_aug2.permute(0, 2, 1)

        pointdict = self.forward(pc_aug1, pc_aug2)

        loss = self.regression_loss(pointdict['qz1_online'], pointdict['z2_target'])
        loss += self.regression_loss(pointdict['qz2_online'], pointdict['z1_target'])

        local_loss = self.CL_N_P_loss(pointdict['f1_online'])
        local_loss += self.CL_N_P_loss(pointdict['f2_online'])
        total_loss = loss + local_loss*0.01
        #
        log = dict(train_loss=total_loss, global_loss=loss, local_loss=local_loss)
        return dict(loss=total_loss, log=log, progress_bar=dict(train_loss=total_loss, global_loss=loss, local_loss=local_loss))

        # log = dict(train_loss=local_loss)
        # return dict(loss=local_loss, log=log, progress_bar=dict(train_loss=local_loss))

    def validation_step(self, batch, batch_idx):
        pc_aug1, pc_aug2 = batch

        if self.hparams["network"] in {"DGCNN", "CurveNet"}:
            pc_aug1 = pc_aug1.permute(0, 2, 1)
            pc_aug2 = pc_aug2.permute(0, 2, 1)

        pointdict = self.forward(pc_aug1, pc_aug2)

 #### try1 with global feature and local feature
        loss = self.regression_loss(pointdict['qz1_online'], pointdict['z2_target'])
        loss += self.regression_loss(pointdict['qz2_online'], pointdict['z1_target'])

#### try2 only with local feature
        local_loss = self.CL_N_P_loss(pointdict['f1_online'])
        local_loss += self.CL_N_P_loss(pointdict['f2_online'])

        total_loss = loss + local_loss*0.01

        log = dict(val_loss=total_loss)
        return dict(val_loss=total_loss)


    def validation_epoch_end(self, outputs):
        reduced_outputs = dict()
        reduced_outputs['val_loss'] = torch.stack([output['val_loss'] for output in outputs]).mean()

        reduced_outputs.update(
            dict(log=reduced_outputs.copy(), progress_bar=reduced_outputs.copy())
        )

        return reduced_outputs

    def configure_optimizers(self):
        if self.hparams["optimizer.type"] == "adam":
            print("Adam optimizer")
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams["optimizer.lr"],
                weight_decay=self.hparams["optimizer.weight_decay"],
            )
            optimizer = LARSWrapper(optimizer)
        elif self.hparams["optimizer.type"] == "adamw":
            print("AdamW optimizer")
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams["optimizer.lr"],
                weight_decay=self.hparams["optimizer.weight_decay"]
            )
        elif self.hparams["optimizer.type"] == "sgd":
            print("SGD optimizer")
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams["optimizer.lr"],
                weight_decay=self.hparams["optimizer.weight_decay"],
                momentum=0.9
            )
        else:
            print("LARS optimizer")
            base_optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams["optimizer.lr"],
                weight_decay=self.hparams["optimizer.weight_decay"],
            )
            optimizer = LARSWrapper(base_optimizer)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams["epochs"], eta_min=0,
                                                                  last_epoch=-1)
        return [optimizer], [lr_scheduler]

    def prepare_data(self):
        train_transforms = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudUpSampling(max_num_points=self.hparams["num_points"] * 2, centroid="random"),
                d_utils.PointcloudRandomCrop(p=0.5, min_num_points=self.hparams["num_points"]),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudRandomCutout(p=0.5, min_num_points=self.hparams["num_points"]),
                d_utils.PointcloudScale(p=1),
                # d_utils.PointcloudRotate(p=1, axis=[0.0, 0.0, 1.0]),
                d_utils.PointcloudRotatePerturbation(p=1),
                d_utils.PointcloudTranslate(p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
                # d_utils.PointcloudSample(num_pt=self.hparams["num_points"])
            ]
        )

        eval_transforms = train_transforms

        train_transforms_scannet_1 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudUpSampling(max_num_points=self.hparams["num_points"] * 2, centroid="random"),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(p=1),
                # d_utils.PointcloudRotate(p=1, axis=np.array([0.0, 0.0, 1.0])),
                d_utils.PointcloudRotatePerturbation(p=1),
                d_utils.PointcloudTranslate(p=1),
                d_utils.PointcloudJitter(p=1),

            ]
        )

        train_transforms_scannet_2 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudRandomCrop(p=0.5, min_num_points=self.hparams["num_points"]),
                d_utils.PointcloudRandomCutout(p=0.5, min_num_points=self.hparams["num_points"]),
                d_utils.PointcloudRandomInputDropout(p=1),
                # d_utils.PointcloudSample(num_pt=self.hparams["num_points"])
            ]
        )

        eval_transforms_scannet_1 = train_transforms_scannet_1
        eval_transforms_scannet_2 = train_transforms_scannet_2

        if self.hparams["dataset"] == "ModelNet40":
            print("Dataset: ModelNet40")
            self.train_dset = ModelNet40ClsContrast(
                self.hparams["num_points"], transforms=train_transforms, train=True
            )


            self.val_dset = ModelNet40ClsContrast(
                self.hparams["num_points"], transforms=eval_transforms, train=False
            )
        elif self.hparams["dataset"] == "ShapeNetPart":
            print("Dataset: ShapeNetPart")
            self.train_dset = PartNormalDatasetContrast(
                self.hparams["num_points"], transforms=train_transforms, split="trainval", normal_channel=True
            )

            self.val_dset = PartNormalDatasetContrast(
                self.hparams["num_points"], transforms=eval_transforms, split="test", normal_channel=True
            )

        elif self.hparams["dataset"] == "ShapeNet":
            print("Dataset: ShapeNet")
            self.train_dset = WholeNormalDatasetContrast(
                self.hparams["num_points"], transforms=train_transforms
            )

            self.val_dset = ModelNet40ClsContrast(
                self.hparams["num_points"], transforms=eval_transforms, train=False, xyz_only=True
            )

        elif self.hparams["dataset"] == "ScanNet":
            print("Dataset: ScanNet")
            self.train_dset = ScannetWholeSceneContrast(
                self.hparams["num_points"], transforms=train_transforms, train=True
            )
            self.val_dset = ModelNet40ClsContrast(
                self.hparams["num_points"], transforms=eval_transforms, train=False, xyz_only=True
            )

        elif self.hparams["dataset"] == "ScanNetFrames":
            print("Dataset: ScanNetFrames")
            self.train_dset = ScanNetFrameContrast(
                self.hparams["num_points"], transforms_1=train_transforms_scannet_1, transforms_2=train_transforms_scannet_2,
                no_height=True, mode=self.hparams["transform_mode"])
            self.val_dset = ScannetWholeSceneContrastHeight(
                self.hparams["num_points"], transforms_1=eval_transforms_scannet_1, transforms_2=eval_transforms_scannet_2, train=False,
                no_height=True)

    def _build_dataloader(self, dset, mode, batch_size=None):
        if batch_size is None:
            batch_size = self.hparams["batch_size"]
        return DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

    def train_dataloader(self):
        train_loader = self._build_dataloader(self.train_dset, mode="train")
        self.epoch_steps = len(train_loader)
        return train_loader

    def val_dataloader(self):
        return self._build_dataloader(self.val_dset, mode="val")


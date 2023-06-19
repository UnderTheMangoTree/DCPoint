"""
@Author: Tiange Xiang
@Contact: txia7609@uni.sydney.edu.au
@File: curvenet_cls.py
@Time: 2021/01/21 3:10 PM
"""

import torch.nn as nn
import torch.nn.functional as F
from .network_curvenet_utils import *
from BYOL.models.utils import sample_and_group


curve_config = {
    'default': [[100, 5], [100, 5], None, None],
    'long': [[10, 30], None, None, None]
}


class CurveNet(nn.Module):
    def __init__(self, hparams):
        super(CurveNet, self).__init__()
        setting = 'default'
        self.hparams = hparams
        k = hparams["k"]
        self.emb_dims = hparams["emb_dims"]


        assert setting in curve_config

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder
        self.cic11 = CIC(npoint=1024, radius=0.05, k=k, in_channels=additional_channel, output_channels=64,
                         bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4,
                         mlp_num=1, curve_config=curve_config[setting][0])

        self.cic21 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2,
                         mlp_num=1, curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=1024, radius=0.1, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4,
                         mlp_num=1, curve_config=curve_config[setting][1])

        self.cic31 = CIC(npoint=256, radius=0.1, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2,
                         mlp_num=1, curve_config=curve_config[setting][2])
        self.cic32 = CIC(npoint=256, radius=0.2, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4,
                         mlp_num=1, curve_config=curve_config[setting][2])

        self.cic41 = CIC(npoint=64, radius=0.2, k=k, in_channels=256, output_channels=512, bottleneck_ratio=2,
                         mlp_num=1, curve_config=curve_config[setting][3])
        self.cic42 = CIC(npoint=64, radius=0.4, k=k, in_channels=512, output_channels=512, bottleneck_ratio=4,
                         mlp_num=1, curve_config=curve_config[setting][3])

        self.conv0 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))
        # self.conv1 = nn.Linear(1024 * 2, 512, bias=False)
        # self.conv2 = nn.Linear(512, num_classes)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=0.5)

    def forward(self, xyz):
        # xyz = xyz.permute(0,2,1)
        batch_size = xyz.size(0)
        num_points = xyz.size(-1)
        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)

        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)

        x = self.conv0(l4_points)
        feats = x.reshape(batch_size, num_points, -1)
        pc = xyz.reshape(batch_size, num_points, -1)

        pcl = sample_and_group(xyz=pc, points=feats, nneiber=self.hparams["nneiber"], seed= self.hparams['seed'])



        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)

        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)  ##object feature SL
        # x = F.relu(self.bn1(self.conv1(x).unsqueeze(-1)), inplace=True).squeeze(-1)
        # x = self.dp1(x)
        # x = self.conv2(x)
        return x, pcl


class TargetNetwork_CurveNet(CurveNet):
    def __init__(self, hparams):
        super(TargetNetwork_CurveNet,self).__init__(hparams)
        self.hparams = hparams
        self.emb_dims = hparams["emb_dims"]

        self.build_target_network()

    def build_target_network(self):
        """
            add a projector MLP to original netwrok
        """
        self.projector = nn.Sequential(
            nn.Linear(self.emb_dims*2, self.hparams["mlp_hidden_size"], bias=False),
            nn.BatchNorm1d(self.hparams["mlp_hidden_size"]),
            nn.ReLU(True),
            nn.Linear(self.hparams["mlp_hidden_size"], self.hparams["projection_size"], bias=False)
        )

    def forward(self, pointcloud):
        """

        :param pointcloud: input point cloud
        :return:
        y: representation
        z: projection
        """
        y,l = super(TargetNetwork_CurveNet, self).forward(pointcloud)
        z = self.projector(y)
        return y, l, z


class OnlineNetwork_CurveNet(TargetNetwork_CurveNet):
    def __init__(self, hparams):
        super(OnlineNetwork_CurveNet, self).__init__(hparams)
        self.hparams = hparams

        self.build_online_network()

    def build_online_network(self):
        """
            add a predictor MLP to target netwrok
        """
        self.predictor = nn.Sequential(
            nn.Linear(self.hparams["projection_size"], self.hparams["mlp_hidden_size"], bias=False),
            nn.BatchNorm1d(self.hparams["mlp_hidden_size"]),
            nn.ReLU(True),
            nn.Linear(self.hparams["mlp_hidden_size"], self.hparams["projection_size"], bias=False)
        )

    def forward(self, pointcloud):
        """

        :param pointcloud: input point cloud
        :return:
        y: representation
        z: projection
        qz: prediction of target network's projection
        """
        y, l, z = super(OnlineNetwork_CurveNet, self).forward(pointcloud)
        qz = self.predictor(z)
        return y, l, z, qz



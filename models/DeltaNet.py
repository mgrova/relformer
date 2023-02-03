import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.TransformerEncoder import TransformerEncoder
from models.transformer.PositionEncoder import PositionEmbeddingLearnedWithPoseToken

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class DeltaRegressor(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, init_stride):
        super().__init__()
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvBnReLU(in_channels, hidden_channels, kernel_size=3, stride=init_stride, pad=1)
        self.conv2 = ConvBn(hidden_channels, hidden_channels, kernel_size=3, stride=1, pad=1)
        self.head = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, delta_img):
        delta_img = self.conv2(self.conv1(delta_img))
        delta_z = self.avg_pooling_2d(delta_img).flatten(start_dim=1)
        delta = self.head(delta_z)
        return delta


class Relformer(nn.Module):

    def __init__(self, hidden_dim, out_dim):
        super().__init__()

        self.transformer_encoder = TransformerEncoder({"hidden_dim":hidden_dim})
        self.position_encoder = PositionEmbeddingLearnedWithPoseToken(hidden_dim//2)
        self.rel_pose_token = nn.Parameter(torch.zeros((1,  hidden_dim)), requires_grad=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, delta_img):
        # make into a sequence and append token
        delta_seq = delta_img.flatten(start_dim=2).permute(2,0,1)
        batch_size = delta_img.shape[0]
        rel_token = self.rel_pose_token.unsqueeze(1).repeat(1, batch_size, 1)
        # S x B x D
        delta_seq = torch.cat([rel_token, delta_seq])

        # prepare position encoding
        token_posisition_encoding, activation_posistion_encoding = self.position_encoder(delta_img)
        position_encoding = torch.cat([token_posisition_encoding.unsqueeze(2).permute(2, 0, 1),
                               activation_posistion_encoding.flatten(2).permute(2, 0, 1)])

        # regress latent relative with transformer encoder
        delta_z = self.transformer_encoder(delta_seq, position_encoding)[:, 0, :]
        delta = self.head(delta_z)
        return delta

class DeltaNet(nn.Module):

    def __init__(self, config, backbone_path):
        super().__init__()

        reduction_map = {"reduction_3": 40, "reduction_4": 112, "reduction_5": 320}
        self.reductions = config.get("reduction")
        self.delta_img_dim = config.get("delta_img_dim")
        self.hidden_dim = config.get("hidden_dim")

        self.proj_x = nn.Conv2d(reduction_map[self.reductions[0]]*2, self.hidden_dim, kernel_size=1)
        self.proj_q = nn.Conv2d(reduction_map[self.reductions[1]]*2, self.hidden_dim, kernel_size=1)

        reg_type = config.get("regressor_type")
        if reg_type == "transformer":
            self.delta_reg_x = Relformer(self.hidden_dim, 3)
            self.delta_reg_q = Relformer(self.hidden_dim, 4)
        elif reg_type == "conv":
            self.delta_reg_x = DeltaRegressor(self.hidden_dim, self.hidden_dim*2, 3, 2)
            self.delta_reg_q = DeltaRegressor(self.hidden_dim, self.hidden_dim * 2, 4, 1)
        else:
            raise NotImplementedError(reg_type)

        self._reset_parameters()

        # load backbone after param reset
        self.double_backbone = config.get("double_backbone")
        if self.double_backbone:
            self.backbone_x = torch.load(backbone_path)
            self.backbone_q = torch.load(backbone_path)
        else:
            self.backbone = torch.load(backbone_path)



    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, data):
        query = data.get('query')
        ref = data.get('ref')

        if self.double_backbone:
            query_endpoints_x = self.backbone_x.extract_endpoints(query)[self.reductions[0]]
            ref_endpoints_x = self.backbone_x.extract_endpoints(ref)[self.reductions[0]]
            query_endpoints_q = self.backbone_q.extract_endpoints(query)[self.reductions[1]]
            ref_endpoints_q = self.backbone_q.extract_endpoints(ref)[self.reductions[1]]
        else:
            query_endpoints = self.backbone.extract_endpoints(query)
            ref_endpoints = self.backbone.extract_endpoints(ref)
            query_endpoints_x = query_endpoints[self.reductions[0]]
            ref_endpoints_x = ref_endpoints[self.reductions[0]]
            query_endpoints_q = query_endpoints[self.reductions[1]]
            ref_endpoints_q = ref_endpoints[self.reductions[1]]

        # delta_img_x is N x 2D x H_R x W_R -- N X 224 x 14 x 14
        delta_img_x = torch.cat((query_endpoints_x, ref_endpoints_x), dim=1)
        # delta_img_q is N x 2D x H_R x W_R  -- N x 80 x 28 x 28
        delta_img_q = torch.cat((query_endpoints_q, ref_endpoints_q), dim=1)

        delta_img_x = self.proj_x(delta_img_x) #N X hidden_dim x 14 x 14
        delta_img_q = self.proj_q(delta_img_q) # #N X hidden_dim x 28 x 28

        delta_x = self.delta_reg_x(delta_img_x)
        delta_q = self.delta_reg_q(delta_img_q)
        delta_p = torch.cat((delta_x, delta_q), dim=1)
        return {"rel_pose": delta_p }





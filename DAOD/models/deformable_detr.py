import math
import copy

import torch
from torch.nn.functional import relu, interpolate, dropout
from torch import nn


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiConv2d(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Conv2d(n, k, kernel_size=(3, 3), padding=1) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiConv1d(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Conv1d(n, k, kernel_size=3) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eta=1.0):
        ctx.eta = eta
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.eta), None


def grad_reverse(x, eta=1.0):
    return GradReverse.apply(x, eta)


class DeformableDETR(nn.Module):

    def __init__(self,
                 backbone,
                 position_encoding,
                 transformer,
                 num_classes=9,
                 num_queries=300,
                 num_feature_levels=4,
                 if_da=False):
        super().__init__()
        # Network hyperparameters
        self.hidden_dim = transformer.hidden_dim
        self.num_feature_levels = num_feature_levels
        self.num_queries = num_queries
        self.num_classes = num_classes
        # Backbone: multiscale outputs backbone network
        self.backbone = backbone
        # Build input projections
        self.input_proj = self._build_input_projections()
        # Position encoding
        self.position_encoding = position_encoding
        # Deformable transformer
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim * 2)
        self.transformer = transformer
        # Prediction of class and box
        self.class_embed = nn.Linear(self.hidden_dim, self.num_classes)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        # domain discriminator
        self.domain_pred_bac, self.domain_pred_enc, self.domain_pred_dec = None, None, None
        self._init_params()
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(transformer.decoder.num_layers)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(transformer.decoder.num_layers)])
        # task 2 domain adaptive module
        if if_da:
            self.da_module = build_da_module()

    def _build_input_projections(self):
        input_proj_list = []
        if self.num_feature_levels > 1:
            for i in range(self.backbone.num_outputs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(self.backbone.num_channels[i], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            in_channels = self.backbone.num_channels[-1]
            for _ in range(self.num_feature_levels - self.backbone.num_outputs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
        else:
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(self.backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                nn.GroupNorm(32, self.hidden_dim),
            ))
        return nn.ModuleList(input_proj_list)

    def _init_params(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)

    def build_discriminators(self, device):
        if self.domain_pred_bac is None and self.domain_pred_enc is None and self.domain_pred_dec is None:
            self.domain_pred_bac = MultiConv2d(self.backbone.num_channels[-1], self.hidden_dim, 2, 3)
            self.domain_pred_bac.to(device)
            self.domain_pred_enc = nn.ModuleList([
                MultiConv2d(self.hidden_dim, self.hidden_dim, 2, 3)
                for _ in range(self.num_feature_levels)
            ])
            self.domain_pred_enc.to(device)
            self.domain_pred_dec = MLP(self.hidden_dim, self.hidden_dim, 2, 3)
            self.domain_pred_dec.to(device)

    @staticmethod
    def inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

    @staticmethod
    def get_mask_list(mask_list, mask_ratio):
        mae_mask_list = copy.deepcopy(mask_list)
        for i in range(len(mask_list)):
            mae_mask_list[i] = torch.rand(mask_list[i].shape).to(mask_list[i].device) < mask_ratio
        return mae_mask_list

    def forward(self, images, masks, enable_mae=False, mask_ratio=0.8, domain_label=None):
        # Backbone forward
        features = self.backbone(images)
        # Prepare input features for transformer
        src_list, mask_list = [], []
        for i, feature in enumerate(features):
            src = self.input_proj[i](feature)
            mask = interpolate(masks[None].float(), size=feature.shape[-2:]).to(torch.bool)[0]
            src_list.append(src)
            mask_list.append(mask)
        if self.num_feature_levels > len(features):
            for i in range(len(features), self.num_feature_levels):
                src = self.input_proj[i](features[-1]) if i == len(features) else self.input_proj[i](src_list[-1])
                mask = interpolate(masks[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                src_list.append(src)
                mask_list.append(mask)
        pos_list = [self.position_encoding(src, mask) for src, mask in zip(src_list, mask_list)]
        query_embeds = self.query_embed.weight
        # Transformer forward
        hs, init_reference, inter_references, _, _, inter_memory, inter_object_query = self.transformer(
            src_list,
            mask_list,
            pos_list,
            query_embeds,
            enable_mae=False,
        )
        # Prepare outputs
        outputs_classes, outputs_coords = [], []
        for lvl in range(hs.shape[0]):
            outputs_class = self.class_embed[lvl](hs[lvl])
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = self.inverse_sigmoid(reference)
            tmp = self.bbox_embed[lvl](hs[lvl])
            tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        out = {
            'logits_all': outputs_class,
            'boxes_all': outputs_coord,
            'features': features[-1].detach()
        }
        # MAE branch
        if enable_mae:
            assert self.transformer.mae_decoder is not None
            mae_mask_list = self.get_mask_list(mask_list, mask_ratio)
            mae_src_list = src_list
            mae_output = self.transformer(
                mae_src_list,
                mae_mask_list,
                pos_list,
                query_embeds,
                enable_mae=enable_mae,
            )
            out['features'] = [features[mae_idx].detach() for mae_idx in [2]]
            out['mae_output'] = mae_output
        # Discriminators
        if self.domain_pred_bac is not None and self.domain_pred_enc is not None and self.domain_pred_dec is not None:
            outputs_domains_bac, outputs_domains_enc, outputs_domains_dec = self.discriminator_forward(
                features, inter_memory, inter_object_query, src_list
            )
            out['domain_bac_all'] = outputs_domains_bac
            out['domain_enc_all'] = outputs_domains_enc
            out['domain_dec_all'] = outputs_domains_dec
        # task 2 的domain adaptive部分
        if domain_label is not None:
            # out['ins_feat'] = hs[-1]
            # out['img_feat'] = src_list
            da_img_features, da_ins_features, da_img_consist_features, da_ins_consist_features, da_label \
                = self.da_module(src_list, hs[-1], domain_label)
            out['da_img_features'] = da_img_features
            out['da_ins_features'] = da_ins_features
            out['da_img_consist_features'] = da_img_consist_features
            out['da_ins_consist_features'] = da_ins_consist_features
        return out

    def discriminator_forward(self, features, inter_memory, inter_object_query, src_list):
        def apply_dis(memory, discriminator):
            return discriminator(grad_reverse(memory))
        
        # Conv discriminator
        outputs_domains_bac = apply_dis(features[-1], self.domain_pred_bac).permute(0, 2, 3, 1)
        sampling_location = 0
        outputs_domains_enc = []
        for lvl, src in enumerate(src_list):
            b, c, h, w = src.shape
            lvl_domains_enc = []
            for hda_idx in range(inter_memory.shape[1]):
                lvl_inter_memory = inter_memory[:, hda_idx, sampling_location: sampling_location + h * w, :]\
                    .transpose(1, 2).reshape(b, c, h, w)  # (b, c, h, w)
                lvl_hda_domains_enc = apply_dis(lvl_inter_memory, self.domain_pred_enc[lvl])  # (b, 2, h, w)
                lvl_hda_domains_enc = lvl_hda_domains_enc.reshape(b, 2, h*w).transpose(1, 2)  # (b, h * w, 2)
                lvl_domains_enc.append(lvl_hda_domains_enc)
            outputs_domains_enc.append(torch.stack(lvl_domains_enc, dim=1))  # (b, hda, h * w, 2)
            sampling_location += h * w
        outputs_domains_enc = torch.cat(outputs_domains_enc, dim=2)
        outputs_domains_dec = apply_dis(inter_object_query, self.domain_pred_dec)
        return outputs_domains_bac, outputs_domains_enc, outputs_domains_dec

# feature alignment的domain adaptive模块
def build_da_module():
    da_module = DomainAdaptationModule()
    return da_module

class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAImgHead, self).__init__()

        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        img_features = []
        for feature in x:
            t = relu(self.conv1_da(feature))
            img_features.append(self.conv2_da(t))
        return img_features


class DAInsHead(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAInsHead, self).__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)

    def forward(self, x):
        x = relu(self.fc1_da(x))
        x = dropout(x, p=0.5, training=self.training)

        x = relu(self.fc2_da(x))
        x = dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x


class DomainAdaptationModule(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals.
    """

    def __init__(self):
        super(DomainAdaptationModule, self).__init__()

        num_ins_inputs = 256

        self.resnet_backbone = True
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

        in_channels = 256

        self.imghead = DAImgHead(in_channels)
        self.inshead = DAInsHead(num_ins_inputs)

    def forward(self, img_features, da_ins_feature, da_label):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors
            da_label (Int): domain label for instance-level feature vectors

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """

        # da_ins_feature = da_ins_feature.view(da_ins_feature.size(0) * da_ins_feature.size(1), -1)
        da_ins_feature = torch.flatten(da_ins_feature, start_dim=0, end_dim=1)

        img_grl_fea = [grad_reverse(fea, 0.1) for fea in img_features]
        ins_grl_fea = grad_reverse(da_ins_feature, 0.1)
        img_grl_consist_fea = [grad_reverse(fea, -0.1) for fea in img_features]
        ins_grl_consist_fea = grad_reverse(da_ins_feature, -0.1)

        da_img_features = self.imghead(img_grl_fea)
        da_ins_features = self.inshead(ins_grl_fea)
        da_img_consist_features = self.imghead(img_grl_consist_fea)
        da_ins_consist_features = self.inshead(ins_grl_consist_fea)
        da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]
        da_ins_consist_features = da_ins_consist_features.sigmoid()

        return da_img_features, da_ins_features, da_img_consist_features, da_ins_consist_features, da_label
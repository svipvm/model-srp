import torch, math
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def fuse_conv_bn(conv, bn):
    temp = bn.weight / (bn.running_var + bn.eps).sqrt()
    if isinstance(conv, torch.Tensor):
        weight = conv * temp.reshape(-1, 1, 1, 1)
        bias = bn.bias - bn.running_mean * temp
    else:
        weight = conv.weight * temp.reshape(-1, 1, 1, 1)
        if isinstance(conv.bias, torch.Tensor):
            bias = (conv.bias - bn.running_mean) * temp + bn.bias
        else:
            bias = bn.bias - bn.running_mean * temp
    return weight, bias

# def fuse_add_branch(branch):
#     weight = sum([b.weight for b in branch])
#     bias = sum([b.bias for b in branch])
#     return weight, bias
def fuse_add_branch(kernels, biases):
    return sum(kernels), sum(biases)

def fuse_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g*k1_group_width: (g+1)*k1_group_width, :, :]
            k2_slice = k2[g * k2_group_width: (g + 1) * k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append((k2_slice * b1[
                g * k1_group_width: (g + 1) * k1_group_width]
                .reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = fuse_depthconcat(k_slices, b_slices)
    return k, b_hat + b2

def fuse_depthconcat(kernels, biases):
    # weight = torch.cat([b.weight for b in branch])
    # bias = torch.cat([b.bias for b in branch])
    # return weight, bias
    return torch.cat(kernels, dim=0), torch.cat(biases)

def fuse_avg(channels, kernel_size, groups):
    input_dim = channels // groups
    weight = torch.zeros(
        (channels, input_dim, kernel_size, kernel_size))
    weight[np.arange(channels), np.tile(np.arange(input_dim), groups)
           , :, :] = 1.0 / (kernel_size ** 2)
    return weight

# This has not been tested with non-square kernels (kernel.size(2) != kernel.size(3)) nor even-size kernels
def fuse_kernel_padding(weight, target_kernel_size: int):
    assert (target_kernel_size - weight.size(2)) % 2 == 0
    assert (target_kernel_size - weight.size(3)) % 2 == 0

    H_to_pad = (target_kernel_size - weight.size(2)) // 2
    W_to_pad = (target_kernel_size - weight.size(3)) // 2
    return F.pad(weight, [H_to_pad, H_to_pad, W_to_pad, W_to_pad])

def add_identity_kxk_kernel(weight, fea_dim, kernel_size=3):
    assert kernel_size % 2 == 1
    weight = weight.clone()
    id_pos = kernel_size // 2
    for i in range(fea_dim):
        weight[i, i, id_pos, id_pos] += 1.0
    return weight

def pad_tensor(t, pattern, padding=1):
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), 'constant', 0)
    t[:, :, 0:padding, :] = pattern
    t[:, :, -padding:, :] = pattern
    t[:, :, :, 0:padding] = pattern
    t[:, :, :, -padding:] = pattern
    return t

class RepConvBNBlock(nn.Module):
    def __init__(self, activate=None, deploy=False, **args):
        super(RepConvBNBlock, self).__init__()
        self.activate = activate
        self.deploy = deploy
        self.args = deepcopy(args)

        if self.deploy:
            self.init_rep_block(**deepcopy(self.args))
        else:
            self.conv = nn.Conv2d(**args)
            self.bn = nn.BatchNorm2d(args['out_channels'])

    def forward(self, inp):
        if self.deploy:
            x = self.rep_conv(inp)
        else:
            x = self.bn(self.conv(inp))

        return getattr(F, self.activate)(x) if self.activate is not None else x

    def init_rep_block(self, **args):
        args['bias'] = True
        self.rep_conv = nn.Conv2d(**args)

    def get_equivalent_kernel_bias(self):
        return fuse_conv_bn(self.conv, self.bn)

    def switch_to_deploy(self):
        if self.deploy: return

        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.__delattr__('conv')
        self.__delattr__('bn')

        self.init_rep_block(**deepcopy(self.args))
        self.rep_conv.weight.data = deploy_k
        self.rep_conv.bias.data = deploy_b
        self.deploy = True

class RepConvPadBNBlock(RepConvBNBlock):
    def __init__(self, activate=None, deploy=False, **args):
        super(RepConvPadBNBlock, self).__init__(
            activate=activate, deploy=deploy, **deepcopy(args))
        assert args.get('kernel_size', 1) == 1
        self.padding = args.get('padding', 0)
        self.deploy = deploy
        self.args = deepcopy(args)

        if self.deploy:
            self.init_rep_block(**deepcopy(self.args))
        else:
            args.update({'padding': 0})
            self.conv = nn.Conv2d(**args)
            self.bn = nn.BatchNorm2d(args['out_channels'])

    def forward(self, inp):
        if self.deploy:
            x = self.rep_conv(inp)
        else:
            x = self.bn(self.conv(inp))
            if self.padding > 0:
                pad_values = self.bn.bias.detach() - \
                    self.bn.running_mean * self.bn.weight.detach() \
                        / torch.sqrt(self.bn.running_var + self.bn.eps)
                pad_values = pad_values.view(1, -1, 1, 1)
                x = F.pad(x, [self.padding] * 4)
                x[:, :, 0:self.padding, :] = pad_values
                x[:, :, -self.padding:, :] = pad_values
                x[:, :, :, 0:self.padding] = pad_values
                x[:, :, :, -self.padding:] = pad_values

        return getattr(F, self.activate)(x) if self.activate is not None else x


class RepIDBasedConv1x1BNBlock(nn.Conv2d):
    def __init__(self, activate=None, deploy=False, **args):
        assert args['in_channels'] == args['out_channels']
        assert args['kernel_size']== 1
        assert args.get('stride', 1) == 1
        self.activate = activate
        self.deploy = deploy
        self.padding_ = args.get('padding', 0)
        args.update({'bias': True})
        self.args = deepcopy(args)

        if self.deploy:
            super(RepIDBasedConv1x1BNBlock, self).__init__(**deepcopy(args))
        else:
            self.groups = args.get('groups', 1)
            channels = args['in_channels']
            assert channels % self.groups == 0
            # args.update({'bais': False})
            super(RepIDBasedConv1x1BNBlock, self).__init__(**deepcopy(args))
            self.bn = nn.BatchNorm2d(channels)

            input_dim = channels // self.groups
            id_value = np.zeros((channels, input_dim, 1, 1))
            for i in range(channels):
                id_value[i, i % input_dim, 0, 0] = 1
            self.id_k = torch.from_numpy(id_value).type_as(self.weight)

    def forward(self, inp):
        if self.deploy:
            x = F.conv2d(inp, self.weight, None, stride=1, padding=0,
                         dilation=self.dilation, groups=self.groups)
        else:
            kernel = self.weight + self.id_k.to(self.weight.device)
            x = F.conv2d(inp, kernel, None, stride=1, padding=0,
                         dilation=self.dilation, groups=self.groups)
            x = self.bn(x)
            if self.padding_ > 0:
                pad_values = self.bn.bias.detach() - \
                    self.bn.running_mean * self.bn.weight.detach() \
                        / torch.sqrt(self.bn.running_var + self.bn.eps)
                pad_values = pad_values.view(1, -1, 1, 1)
                x = F.pad(x, [self.padding_] * 4)
                x[:, :, 0:self.padding_, :] = pad_values
                x[:, :, -self.padding_:, :] = pad_values
                x[:, :, :, 0:self.padding_] = pad_values
                x[:, :, :, -self.padding_:] = pad_values

        return getattr(F, self.activate)(x) if self.activate is not None else x

    def get_equivalent_kernel_bias(self):
        return fuse_conv_bn(
            self.weight + self.id_k.to(self.weight.device), self.bn)

    def switch_to_deploy(self):
        if self.deploy: return

        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.__delattr__('bn')

        self.weight.data = deploy_k
        self.bias.data = deploy_b
        self.deploy = True

class ACBlock(nn.Module):
    def __init__(self, activate=None, deploy=False, **args):
        super(ACBlock, self).__init__()
        self.activate = activate
        self.deploy = deploy
        self.args = deepcopy(args)

        if self.deploy:
            self.init_rep_block(**deepcopy(self.args))
        else:
            padding, kernel_size = args['padding'], args['kernel_size']
            if padding - kernel_size // 2 >= 0:
                #   Common use case. E.g., k=3, p=1 or k=5, p=2
                self.crop = 0
                #   Compared to the KxK layer, the padding of the 1xK layer and Kx1 layer should be adjust to align the sliding windows (Fig 2 in the paper)
                hor_padding = [padding - kernel_size // 2, padding]
                ver_padding = [padding, padding - kernel_size // 2]
            else:
                #   A negative "padding" (padding - kernel_size//2 < 0, which is not a common use case) is cropping.
                #   Since nn.Conv2d does not support negative padding, we implement it manually
                self.crop = kernel_size // 2 - padding
                hor_padding = [0, padding]
                ver_padding = [padding, 0]

            args['bias'] = False
            self.square_conv = RepConvBNBlock(**deepcopy(args))
            args['kernel_size'] = (kernel_size, 1)
            args['padding'] = ver_padding
            self.ver_conv = RepConvBNBlock(**deepcopy(args))
            args['kernel_size'] = (1, kernel_size)
            args['padding'] = hor_padding
            self.hor_conv = RepConvBNBlock(**deepcopy(args))

    def init_rep_block(self, **args):
        args['bias'] = True
        self.rep_conv = nn.Conv2d(**args)

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h, asym_w = asym_kernel.shape[2:]
        square_h, square_w = square_kernel.shape[2:]
        square_kernel[:, :, square_h // 2 - asym_h // 2:
                            square_h // 2 - asym_h // 2 + asym_h,
                            square_w // 2 - asym_w // 2:
                            square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

    def get_equivalent_kernel_bias(self):
        self.square_conv.switch_to_deploy()
        self.hor_conv.switch_to_deploy()
        self.ver_conv.switch_to_deploy()

        self._add_to_square_kernel(self.square_conv.rep_conv.weight.data,
                                   self.hor_conv.rep_conv.weight.data)
        self._add_to_square_kernel(self.square_conv.rep_conv.weight.data,
                                   self.ver_conv.rep_conv.weight.data)
        return (self.square_conv.rep_conv.weight,
                    self.square_conv.rep_conv.bias +
                    self.hor_conv.rep_conv.bias +
                    self.ver_conv.rep_conv.bias)

    def switch_to_deploy(self):
        if self.deploy: return

        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.__delattr__('square_conv')
        self.__delattr__('hor_conv')
        self.__delattr__('ver_conv')

        self.init_rep_block(**deepcopy(self.args))
        self.rep_conv.weight.data = deploy_k
        self.rep_conv.bias.data = deploy_b
        self.deploy = True

    def forward(self, inp):
        if self.deploy:
            x = self.rep_conv(inp)
        else:
            square_outputs = self.square_conv(inp)
            if self.crop > 0:
                ver_input = inp[:, :, :, self.crop:-self.crop]
                hor_input = inp[:, :, self.crop:-self.crop, :]
            else:
                ver_input = inp
                hor_input = inp
            vertical_outputs = self.ver_conv(ver_input)
            horizontal_outputs = self.hor_conv(hor_input)
            x = square_outputs + vertical_outputs + horizontal_outputs

        return getattr(F, self.activate)(x) if self.activate is not None else x

class RepVGGBlock(nn.Module):
    def __init__(self, activate=None, deploy=False, **args):
        super(RepVGGBlock, self).__init__()
        self.activate = activate
        self.deploy = deploy
        self.args = deepcopy(args)

        # must be kernel_size=3, padding=1, stride=1
        assert 'kernel_size' not in args or args['kernel_size'] == 3
        assert 'padding' in args and args['padding'] == 1
        assert 'stride' not in args or args['stride'] == 1
        assert args['in_channels'] == args['out_channels']

        if self.deploy:
            self.init_rep_block(**deepcopy(self.args))
        else:
            self.vgg_identity = nn.BatchNorm2d(num_features=args['in_channels'])
            args['bias'] = False
            self.vgg_dense = RepConvBNBlock(**deepcopy(args))
            args['kernel_size'] = 1
            args['padding'] = 0
            self.vgg_1x1 = RepConvBNBlock(**deepcopy(args))

    def init_rep_block(self, **args):
        args['bias'] = True
        self.rep_conv = nn.Conv2d(**args)

    def _fuse_bn_tensor(self, bn):
        if 'groups' not in self.args:
            self.args['groups'] = 1

        input_dim = self.args['in_channels'] // self.args['groups']
        kernel_value = np.zeros(
            (self.args['in_channels'], input_dim, 3, 3), dtype=np.float32)
        for i in range(self.args['in_channels']):
            kernel_value[i, i % input_dim, 1, 1] = 1
        kernel = torch.from_numpy(kernel_value).to(bn.weight.device)

        temp = bn.weight / (bn.running_var + bn.eps).sqrt()
        weight = kernel * temp.reshape(-1, 1, 1, 1)
        bias = bn.bias - bn.running_mean * temp
        return weight, bias

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def get_equivalent_kernel_bias(self):
        id_weight, id_bias = self._fuse_bn_tensor(self.vgg_identity)
        self.vgg_dense.switch_to_deploy()
        self.vgg_1x1.switch_to_deploy()

        return (self.vgg_dense.rep_conv.weight +
                    self._pad_1x1_to_3x3_tensor(self.vgg_1x1.rep_conv.weight) +
                    id_weight,
                self.vgg_dense.rep_conv.bias +
                    self.vgg_1x1.rep_conv.bias +
                    id_bias)

    def switch_to_deploy(self):
        if self.deploy: return

        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.__delattr__('vgg_identity')
        self.__delattr__('vgg_dense')
        self.__delattr__('vgg_1x1')

        self.init_rep_block(**deepcopy(self.args))
        self.rep_conv.weight.data = deploy_k
        self.rep_conv.bias.data = deploy_b
        self.deploy = True

    def forward(self, inp):
        if self.deploy:
            x = self.rep_conv(inp)
        else:
            x = self.vgg_identity(inp) + \
                self.vgg_dense(inp) + \
                self.vgg_1x1(inp)

        return getattr(F, self.activate)(x) if self.activate is not None else x

class DiverseBranchBlock(nn.Module):
    def __init__(self, activate=None, deploy=False, **args):
        super(DiverseBranchBlock, self).__init__()
        self.activate = activate
        self.deploy = deploy
        self.args = deepcopy(args)

        self.in_channels = args['in_channels']
        self.out_channels = args['out_channels']
        self.kernel_size = args['kernel_size']
        self.padding = args.get('padding', self.kernel_size // 2)
        self.stride = args.get('stride', 1)
        self.groups = args.get('groups', 1)
        assert self.padding == self.kernel_size // 2
        assert self.groups != self.out_channels or \
            self.in_channels == self.out_channels

        if self.deploy:
            args.update({"padding": self.padding})
            self.init_rep_block(**args)
        else:
            self.depthwise = (self.out_channels == self.groups)
            self.dbb_origin = RepConvBNBlock(**deepcopy(args))

            if self.depthwise:  # depthwise channel
                self.dbb_avg = nn.Sequential(
                    nn.AvgPool2d(self.kernel_size, self.stride, self.padding),
                    nn.BatchNorm2d(self.out_channels)
                )
            else:
                args.update({'kernel_size': 1, 'padding': 0, 'bias': False})
                self.dbb_1x1 = RepConvBNBlock(**deepcopy(args))
                args.update({'stride': 1, 'padding': self.padding})
                self.dbb_avg = nn.Sequential(
                    RepConvPadBNBlock(**deepcopy(args)),
                    nn.AvgPool2d(self.kernel_size, self.stride, 0),
                    nn.BatchNorm2d(self.out_channels)
                )

            # For mobilenet, it is better to have 2X internal channels
            internal_channels_1x1_3x3 = \
                2 * self.in_channels if self.depthwise else self.in_channels
            dbb_kxk_type = []
            args.update({
                'out_channels': internal_channels_1x1_3x3,
                'kernel_size': 1,
                'stride': 1,
                'padding': self.padding})
            if self.depthwise:
                dbb_kxk_type = [RepConvPadBNBlock(**deepcopy(args))]
            else:
                dbb_kxk_type = [RepIDBasedConv1x1BNBlock(**deepcopy(args))]

            args.update({
                'in_channels': internal_channels_1x1_3x3,
                'out_channels': self.out_channels,
                'kernel_size': self.kernel_size,
                'stride': self.stride,
                'padding': 0})
            dbb_kxk_type += [RepConvBNBlock(**deepcopy(args))]
            self.dbb_kxk = nn.Sequential(*dbb_kxk_type)

    def get_equivalent_kernel_bias(self):
        self.dbb_origin.switch_to_deploy()
        origin_weight = self.dbb_origin.rep_conv.weight
        origin_bias = self.dbb_origin.rep_conv.bias

        if not self.depthwise:
            self.dbb_1x1.switch_to_deploy()
            dbb_1x1_weight = fuse_kernel_padding(
                self.dbb_1x1.rep_conv.weight, self.args['kernel_size'])
            dbb_1x1_bais = self.dbb_1x1.rep_conv.bias
        else:
            dbb_1x1_weight, dbb_1x1_bais = 0, 0

        kernel_avg = fuse_avg(self.out_channels, self.kernel_size, self.groups)
        avg_weight, avg_bias = fuse_conv_bn(
            kernel_avg.to(self.dbb_avg[-1].weight.device), self.dbb_avg[-1])
        if not self.depthwise:
            self.dbb_avg[0].switch_to_deploy()
            avg_weight, avg_bias = fuse_1x1_kxk(self.dbb_avg[0].rep_conv.weight,
                         self.dbb_avg[0].rep_conv.bias,
                         avg_weight, avg_bias, self.groups)

        assert len(self.dbb_kxk) == 2
        self.dbb_kxk[0].switch_to_deploy()
        if hasattr(self.dbb_kxk[0], 'rep_conv'):
            weight_kxk_1th = self.dbb_kxk[0].rep_conv.weight
            bais_kxk_1th = self.dbb_kxk[0].rep_conv.bias
        else:
            weight_kxk_1th = self.dbb_kxk[0].weight
            bais_kxk_1th = self.dbb_kxk[0].bias
        self.dbb_kxk[1].switch_to_deploy()
        weight_kxk_2th = self.dbb_kxk[1].rep_conv.weight
        bais_kxk_2th = self.dbb_kxk[1].rep_conv.bias
        kxk_weight, kxk_bias = fuse_1x1_kxk(weight_kxk_1th, bais_kxk_1th,
            weight_kxk_2th, bais_kxk_2th, groups=self.groups)

        return fuse_add_branch(
            (origin_weight, dbb_1x1_weight, avg_weight, kxk_weight),
            (origin_bias, dbb_1x1_bais, avg_bias, kxk_bias))
            # (origin_weight, dbb_1x1_weight, avg_weight),
            # (origin_bias, dbb_1x1_bais, avg_bias))

    def init_rep_block(self, **args):
        args['bias'] = True
        self.rep_conv = nn.Conv2d(**args)

    def switch_to_deploy(self):
        if self.deploy: return

        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.__delattr__('dbb_origin')
        if not self.depthwise:
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_avg')
        self.__delattr__('dbb_kxk')

        self.init_rep_block(**deepcopy(self.args))
        self.rep_conv.weight.data = deploy_k
        self.rep_conv.bias.data = deploy_b
        self.deploy = True

    def forward(self, inp):
        if self.deploy:
            x = self.rep_conv(inp)
        else:
            x = self.dbb_origin(inp)
            if not self.depthwise:
                x += self.dbb_1x1(inp)
            x += self.dbb_avg(inp)
            x += self.dbb_kxk(inp)

        return getattr(F, self.activate)(x) if self.activate is not None else x

class ResRepBlock(nn.Module):
    """ Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.

    Diagram:
        ---Conv1x1--Conv3x3-+-Conv1x1--+--
                   |________|
         |_____________________________|
    """
    def __init__(self, activate=None, deploy=False, expand_factor=2, **args):
        super(ResRepBlock, self).__init__()
        self.activate = activate
        self.deploy = deploy
        args.update({'bias': True})
        self.args = deepcopy(args)

        assert args['in_channels'] == args['out_channels']
        assert args['kernel_size'] == 3
        assert args.get('stride', 1) == 1
        assert args.get('padding', 0) == 1
        channels = args['in_channels']
        self.padding = args.get('padding', 1)

        if self.deploy:
            self.init_rep_block(**deepcopy(self.args))
        else:
            args.update({
                'out_channels': int(args['in_channels']) * expand_factor,
                'kernel_size': 1,
                'padding': 0
            })
            self.expand_conv = nn.Conv2d(**deepcopy(args))
            args.update({
                'in_channels': int(args['in_channels']) * expand_factor,
                'kernel_size': 3
            })
            self.fea_conv = nn.Conv2d(**deepcopy(args))
            args.update({
                'out_channels': channels,
                'kernel_size': 1
            })
            self.reduce_conv = nn.Conv2d(**deepcopy(args))

    def init_rep_block(self, **args):
        args['bias'] = True
        self.rep_conv = nn.Conv2d(**args)

    def get_equivalent_kernel_bias(self):
        k0 = self.expand_conv.weight
        b0 = self.expand_conv.bias
        expand_dim, in_dim = k0.shape[:2]

        k1 = self.fea_conv.weight
        b1 = self.fea_conv.bias

        k2 = self.reduce_conv.weight
        b2 = self.reduce_conv.bias

         # first step: remove the middle identity
        identity_k1 = add_identity_kxk_kernel(k1, expand_dim)

        # second step: merge the first 1x1 convolution and the next 3x3 convolution
        weight_k0k1 = F.conv2d(input=identity_k1, weight=k0.permute(1, 0, 2, 3))
        bias_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(
            1, expand_dim, 3, 3).to(self.fea_conv.weight.device)
        bias_b0b1 = F.conv2d(input=bias_b0b1, weight=identity_k1, bias=b1)

        # third step: merge the remain 1x1 convolution
        weight_k0k1k2 = F.conv2d(input=weight_k0k1.permute(1, 0, 2, 3),
            weight=k2).permute(1, 0, 2, 3)
        bias_k0k1k2 = F.conv2d(input=bias_b0b1, weight=k2, bias=b2).view(-1)

        # last step: remove the global identity
        identity_k0k1k2 = add_identity_kxk_kernel(weight_k0k1k2, in_dim)

        return (identity_k0k1k2, bias_k0k1k2)

    def switch_to_deploy(self):
        if self.deploy: return

        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.__delattr__('expand_conv')
        self.__delattr__('fea_conv')
        self.__delattr__('reduce_conv')

        self.init_rep_block(**deepcopy(self.args))
        self.rep_conv.weight.data = deploy_k
        self.rep_conv.bias.data = deploy_b
        self.deploy = True

    def forward(self, inp):
        if self.deploy:
            x = self.rep_conv(inp)
        else:
            x = self.expand_conv(inp)
            out_identity = x

            b0 = self.expand_conv.bias
            x = pad_tensor(x, b0)

            x = self.fea_conv(x) + out_identity
            x = self.reduce_conv(x)
            x += inp

        return getattr(F, self.activate)(x) if self.activate is not None else x

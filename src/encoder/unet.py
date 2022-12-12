'''
Codes are from:
https://github.com/jaxony/unet-pytorch/blob/master/model.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
import copy

from src.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate, map2local
from torch_scatter import scatter_mean, scatter_max

hidden_size = 32 
depth = 4 # 3 # 4 #4 # 3

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, i, pooling):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.downsample = i #False # pooling
        # print('**********self.pooling', i)
        # print('self.inchanens', self.in_channels)
        # print('self.outchannels', self.out_channels)

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)

        # if self.pooling:
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        
        print('in_channels_downsample', in_channels)
        print('out_channels', out_channels)
        
        # self.fc_comm = nn.Sequential(
        #          nn.Linear(hidden_size,2*hidden_size),
        #          nn.ReLU(),
        #          nn.Linear(2*hidden_size, hidden_size)
        #          ) ###
        # self.fc_comm_list = []
        # for i in range(depth):
        #    hidden_size = 32 * 2 ** i
        self.fc_comm = nn.Sequential(
                    nn.Linear(out_channels,2*out_channels),
                    nn.ReLU(),
                    # nn.Linear(2*out_channels, 3*out_channels),
                    # nn.ReLU(),
                    # nn.Linear(3*out_channels, 2*out_channels),
                    # nn.ReLU(),
                    nn.Linear(2*out_channels, out_channels)
                    ) ###
        #    self.fc_comm_list.append(self.fc_comm)

        # self.fc_c = [nn.Linear(32 * 2 ** i, 32 * 2 ** (i+1)).cuda() for i in range(depth)]
        self.fc_c = nn.Linear(in_channels, out_channels)

        self.padding = 0.1 
        self.reso_plane = 128
        self.c_dim = 32 
        self.sample_mode = 'bilinear'
        
        
        # self.conv1x1 = [conv1x1(32 * 2 ** i, 32 * 2 ** (i+1)).cuda() for i in range(depth)]
        if i > 0:
            self.conv1x1 = conv1x1(in_channels, out_channels)



    def generate_plane_features(self, p, c, channel, reso_plane, plane='xz'):
       
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(xy, reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), channel, reso_plane**2)
        c = c.permute(0, 2, 1) # B x 512 x T

        fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), channel, reso_plane, reso_plane) # sparce matrix (B x 512 x reso x reso)

        '''
        if loop_num == 1:
            # fea_plane = self.unet_loop1(fea_plane+c_plane[plane]) # skip connection between output of cnn to the input of next 2d-cnn 
            # return fea_plane 
            return self.unet_loop1(fea_plane) + c_plane[plane] # as jj sugggested, skip connection between output of cnn to the OUTPUT of next 2d-cnn

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)
        '''
        
        return fea_plane

    def sample_plane_feature(self, p, c, plane='xz'):
        # print('pppppp', p.shape)
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        # print('xy', xy.shape)
        xy = xy[:, :, None].float()
        # print('xy', xy.shape)
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        # print('c====', c.shape)
        return c

    def forward(self, p, x, plane_type, x_after_conv=None, c_last=None):
        # print('xxy', x['xy'].shape)
        # print('**********self.pooling', self.downsample)
        x['xy'] = F.relu(self.conv1(x['xy']))
        x['xy'] = F.relu(self.conv2(x['xy'])) # 
        
        x['xz'] = F.relu(self.conv1(x['xz']))
        x['xz'] = F.relu(self.conv2(x['xz']))

        
        x['yz'] = F.relu(self.conv1(x['yz']))
        x['yz'] = F.relu(self.conv2(x['yz']))

        channel = x['yz'].shape[1]
        reso_plane = x['yz'].shape[2]

        # print('channel', channel)


        if x_after_conv == None:
            x_after_conv = x_after_conv
        else:
            # print('x_after_conv', x_after_conv['yz'].shape)
            # print('===========x', x['yz'].shape)
            # x['yz'] = x['yz'] + self.conv1x1[int(np.log2(channel//32))-1](self.pool(x_after_conv['yz']))
            # x['xy'] = x['xy'] + self.conv1x1[int(np.log2(channel//32))-1](self.pool(x_after_conv['xy']))
            # x['xz'] = x['xz'] + self.conv1x1[int(np.log2(channel//32))-1](self.pool(x_after_conv['xz']))
            # print('============self.pooling', self.downsample)
            if self.downsample in [2, 3]:
                x['yz'] = x['yz'] + self.conv1x1(self.pool(x_after_conv['yz'])) #
                x['xy'] = x['xy'] + self.conv1x1(self.pool(x_after_conv['xy']))
                x['xz'] = x['xz'] + self.conv1x1(self.pool(x_after_conv['xz']))
     
            
            else:
                x['yz'] = x['yz'] + self.conv1x1(x_after_conv['yz']) # self.pool(
                x['xy'] = x['xy'] + self.conv1x1(x_after_conv['xy'])
                x['xz'] = x['xz'] + self.conv1x1(x_after_conv['xz'])

        


        x_after_conv = {}
        x_after_conv['xz'] = x['xz']
        x_after_conv['xy'] = x['xy']
        x_after_conv['yz'] = x['yz']
        # print('xxy', x['xy'].shape)
        
        # if self.c_dim != 0:
        #    plane_type = list(c_plane.keys())
        #    c = 0
        #    if 'grid' in plane_type:
        #        c += self.sample_grid_feature(p, c_plane['grid'])
        c = 0
        if 'xz' in plane_type:
            c += self.sample_plane_feature(p, x['xz'], plane='xz')
        if 'xy' in plane_type:
            c += self.sample_plane_feature(p, x['xy'], plane='xy')
        if 'yz' in plane_type:
            c += self.sample_plane_feature(p, x['yz'], plane='yz')
        c = c.transpose(1, 2)

        # print('cccccc', c.shape)
        # print(channel//32)
        # print('ccccccccccccc', c.shape)

        # c = self.fc_comm_list[int(np.log2(channel//32))](c)

        # print('ccccccccccccc', c.shape)
        c = self.fc_comm(c)
        

        if c_last == None:
            c = c
        else:
        #    c = c + self.fc_c[int(np.log2(channel//32))-1](c_last)
            c = c + self.fc_c(c_last)

        if 'xz' in plane_type:
            x['xz'] = self.generate_plane_features(p, c,  channel, reso_plane, plane='xz') # [1, 32, 64, 64]
            # print('fea[xz]', fea['xz'].shape)
        if 'xy' in plane_type:
            x['xy'] = self.generate_plane_features(p, c,channel, reso_plane, plane='xy')
        if 'yz' in plane_type:
            x['yz'] = self.generate_plane_features(p, c,channel, reso_plane, plane='yz')

        # print('xxy', x['xy'].shape)
        

        before_pool = {}
        before_pool['xz'] = x['xz']
        before_pool['xy'] = x['xy']
        before_pool['yz'] = x['yz']
       

        if self.pooling:
            x['xz'] = self.pool(x['xz'])
            x['xy'] = self.pool(x['xy'])
            x['yz'] = self.pool(x['yz'])
            # print('x[xz]', x['xz'].shape)

        # print('XXXXXXXXXXXXXXXXX', x['xz'].shape)

        # print('before_poolllllllll', before_pool['xy'].shape)
        return x, before_pool, x_after_conv, c


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, i,
                 merge_mode='concat', up_mode='transpose'):

        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
           mode=self.up_mode)
        if i == 2:
            self.upconv_noup = conv1x1(self.in_channels, self.out_channels)

        print(in_channels)
        self.in_channels = in_channels
        print('out_channels', out_channels)

        '''
        self.fc_comm_list = []
        for i in range(depth):
            hidden_size = 32 * 2 ** i
            self.fc_comm = nn.Sequential(
                     nn.Linear(hidden_size,2*hidden_size),
                     nn.ReLU(),
                     # nn.Linear(2*hidden_size, 3*hidden_size),
                     # nn.ReLU(),
                     # nn.Linear(3*hidden_size, 2*hidden_size),
                     # nn.ReLU(),
                     nn.Linear(2*hidden_size, hidden_size)
                     ).cuda() ###
            self.fc_comm_list.append(self.fc_comm)
        '''
        self.fc_comm = nn.Sequential(
            nn.Linear(out_channels, 2*out_channels),
            nn.ReLU(),
            # nn.Linear(2*out_channels, 3*out_channels),
            # nn.ReLU(),
            # nn.Linear(3*out_channels, 2*out_channels),
            # nn.ReLU(),
            nn.Linear(2*out_channels, out_channels)
            ) ###

        # self.fc_c = [nn.Linear(32 * 2 ** (i+1), 32 * 2 ** i).cuda() for i in range(depth)]
        self.fc_c = nn.Linear(in_channels, out_channels)


        # self.conv1x1 = [upconv2x2(32 * 2 ** (i+1), 32 * 2 ** i, mode=self.up_mode).cuda() for i in range(depth)]

        # self.conv1x1 = [conv1x1(32 * 2 ** (i+1), 32 * 2 ** i).cuda() for i in range(depth)]
        if i == 2:
            self.conv1x1 = conv1x1(in_channels, out_channels)
        else:
            self.conv1x1 = upconv2x2(in_channels, out_channels, mode=self.up_mode)
        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)

        self.padding = 0.1 
        self.reso_plane = 128
        self.c_dim = 32 
        self.sample_mode = 'bilinear'
    
    def generate_plane_features(self, p, c, channel, reso_plane, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(xy, reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), channel, reso_plane**2)
        c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), channel, reso_plane, reso_plane) # sparce matrix (B x 512 x reso x reso)

        '''
        if loop_num == 1:
            # fea_plane = self.unet_loop1(fea_plane+c_plane[plane]) # skip connection between output of cnn to the input of next 2d-cnn 
            # return fea_plane 
            return self.unet_loop1(fea_plane) + c_plane[plane] # as jj sugggested, skip connection between output of cnn to the OUTPUT of next 2d-cnn

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)
        '''
        
        return fea_plane

    def sample_plane_feature(self, p, c, plane='xz'):
        # print('pppppp', p.shape)
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        # print('xy', xy.shape)
        xy = xy[:, :, None].float()
        # print('xy', xy.shape)
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        # print('c====', c.shape)
        return c

    def forward(self, p, from_down, from_up, plane_type, x_after_conv, c_last, i):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway,
            from_up: upconv'd tensor from the decoder pathway
        """


        if i == 2:
            from_up['xy'] = self.upconv_noup(from_up['xy'])
            from_up['xz'] = self.upconv_noup(from_up['xz'])
            from_up['yz'] = self.upconv_noup(from_up['yz'])
        else:
            from_up['xy'] = self.upconv(from_up['xy'])
            from_up['xz'] = self.upconv(from_up['xz'])
            from_up['yz'] = self.upconv(from_up['yz'])
        # print('from_up', from_up['xy'].shape)

        # print('from_down', from_down['xy'].shape)

        x = {}
        if self.merge_mode == 'concat':
             x['xy'] = torch.cat((from_up['xy'], from_down['xy']), 1)
             x['xz'] = torch.cat((from_up['xz'], from_down['xz']), 1)
             x['yz'] = torch.cat((from_up['yz'], from_down['yz']), 1)
        else:
            x['xy'] = from_up['xy'] + from_down['xy']
            x['xz'] = from_up['xz'] + from_down['xz']
            x['yz'] = from_up['yz'] + from_down['yz']

        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))

        x['xy'] = F.relu(self.conv1(x['xy']))
        x['xy'] = F.relu(self.conv2(x['xy'])) # F.relu(
        
        x['xz'] = F.relu(self.conv1(x['xz']))
        x['xz'] = F.relu(self.conv2(x['xz']))

        
        x['yz'] = F.relu(self.conv1(x['yz']))
        x['yz'] = F.relu(self.conv2(x['yz'])) # 

        channel = x['yz'].shape[1]
        # print('xyz', x['yz'].shape)
        # print('channel', channel)
        reso_plane = x['yz'].shape[2]
        # print('xyzzzzzz', x['yz'].shape)

        # print(x_after_conv['yz'].shape)
        
        
        if x_after_conv == None:
            x_after_conv = x_after_conv
        else:
            # print('x_after_conv', x_after_conv['yz'].shape)
            # x['yz'] = x['yz'] + self.conv1x1[int(np.log2(channel//32))](x_after_conv['yz'])
            # x['xy'] = x['xy'] + self.conv1x1[int(np.log2(channel//32))](x_after_conv['xy'])
            # x['xz'] = x['xz'] + self.conv1x1[int(np.log2(channel//32))](x_after_conv['xz'])
            x['yz'] = x['yz'] + self.conv1x1(x_after_conv['yz'])
            x['xy'] = x['xy'] + self.conv1x1(x_after_conv['xy'])
            x['xz'] = x['xz'] + self.conv1x1(x_after_conv['xz'])


        x_after_conv = {}
        x_after_conv['xz'] = x['xz']
        x_after_conv['xy'] = x['xy']
        x_after_conv['yz'] = x['yz']

        if i == depth - 2:
            # print('output===', x['yz'].shape)
            return x, x_after_conv, c_last 

        c = 0
        if 'xz' in plane_type:
            c += self.sample_plane_feature(p, x['xz'], plane='xz')
        if 'xy' in plane_type:
            c += self.sample_plane_feature(p, x['xy'], plane='xy')
        if 'yz' in plane_type:
            c += self.sample_plane_feature(p, x['yz'], plane='yz')
        c = c.transpose(1, 2)

        # print('ccccccc', c.shape)

        #c = self.fc_comm_list[int(np.log2(channel//32))](c)
        c = self.fc_comm(c)

        if c_last == None:
            c = c
        else:
            # c = c + self.fc_c[int(np.log2(channel//32))](c_last)
            # print('cccccccafter', c.shape)
            c = c + self.fc_c(c_last)

        if 'xz' in plane_type:
            x['xz'] = self.generate_plane_features(p, c, channel, reso_plane, plane='xz') # [1, 32, 64, 64]
            # print('fea[xz]', x['xz'].shape)
        if 'xy' in plane_type:
            x['xy'] = self.generate_plane_features(p, c, channel, reso_plane, plane='xy')
            # print('fea[xy]', x['xy'].shape)
        if 'yz' in plane_type:
            x['yz'] = self.generate_plane_features(p, c, channel, reso_plane, plane='yz')
            # print('fea[yz]', x['yz'].shape)
        

        return x, x_after_conv, c # c_last 


class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=5, 
                 start_filts=64, up_mode='transpose', 
                 merge_mode='concat', **kwargs):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # merge_mode = 'add'
        
        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            # pooling = True if i < depth-1 else False
            if i == 0 or i == depth-1:
                pooling = False
            else:
                pooling = True
            print('ins', ins)
            print('outs', outs)
            print(pooling)
            down_conv = DownConv(ins, outs, i, pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, i, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        # print(self.down_convs)
        # print(self.up_convs)

        self.conv_final = conv1x1(outs, self.num_classes)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, p, x, plane_type, c):
        encoder_outs = []
        x_after_conv = None
        # print(plane_type)
        # c = None
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            # print('11111111111111111111', i)
            x, before_pool, x_after_conv, c = module(p, x, plane_type, x_after_conv, c)
            # print('forawrdxxxx', x['xz'].shape)
            # print('before_pool_before', before_pool['xy'].shape)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            # print('module', module)
            before_pool = encoder_outs[-(i+2)]
            # print('beforepool', before_pool['xy'].shape)
            x, x_after_conv, c = module(p, before_pool, x, plane_type, x_after_conv, c, i)
        
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x['xz'] = self.conv_final(x['xz'])
        x['xy'] = self.conv_final(x['xy'])
        x['yz'] = self.conv_final(x['yz'])

        return x

if __name__ == "__main__":
    """
    testing
    """
    model = UNet(1, depth=5, merge_mode='concat', in_channels=1, start_filts=32)
    print(model)
    print(sum(p.numel() for p in model.parameters()))

    reso = 176
    x = np.zeros((1, 1, reso, reso))
    x[:,:,int(reso/2-1), int(reso/2-1)] = np.nan
    x = torch.FloatTensor(x)

    out = model(x)
    print('%f'%(torch.sum(torch.isnan(out)).detach().cpu().numpy()/(reso*reso)))
    
    # loss = torch.sum(out)
    # loss.backward()

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import ResnetBlockFC
from src.common import normalize_coordinate, normalize_3d_coordinate, map2local
from torch import distributions as dist
# from src.transformer import *
from src.attention import *
import numpy as np
import math

class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128, num_iterations=2,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.num_iterations = num_iterations
        
        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])
        '''
        self.fc_comm = nn.Sequential(
                 nn.Linear(hidden_size,2*hidden_size),
                 nn.ReLU(),
                 nn.Linear(2*hidden_size, hidden_size)
                 ) ###
        
        # self.fc_comm = ResnetBlockFC(hidden_size, hidden_size, 2*hidden_size)

        self.fc_comm_loop1 = nn.Sequential(
                nn.Linear(hidden_size,2*hidden_size),
                nn.ReLU(),
                nn.Linear(2*hidden_size, hidden_size)
                ) ###       
        '''

        # self.fc_comm = nn.Linear(hidden_size, hidden_size) ###
        
        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
    

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c


    def forward(self, p, c_plane, inputs=None, point_features=None):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_grid_feature(p, c_plane['grid'])
            if 'xz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
            c = c.transpose(1, 2)
            
            
        # print('c_decoder', c.shape) # [1, 2048, 32]
        # c = self.fc_comm(c)
        
        # print('ccccccc', c.shape)

        # print('pointfeature', point_features.shape)

        # if loop_num == 0:
        #    c = self.fc_comm(c) # zw moved here 8/6
        # else:
        #    c = self.fc_comm_loop1(c)

        ###
        # if point_features == None:
        #     c = c
        # else:
        #     c = c + point_features

        ###
        # if loop_num == self.num_iterations  - 1: 
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)
    
        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
            
        p_r = dist.Bernoulli(logits=out)
        return p_r



############################ substract + positional encoding attention ###################
class LocalDecoder_attention_sub(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128, num_iterations=2,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, **decoder_kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.num_iterations = num_iterations
        self.plane_type = decoder_kwargs['plane_type']

        print('c_dim=', c_dim)

        ############################ 3d feature volume #############################
        if self.plane_type == 'grid':
            num_heads = decoder_kwargs['num_heads']
            hidden_size = hidden_size * num_heads
            self.fc_q = nn.Linear(c_dim, hidden_size)
            self.fc_k = nn.Linear(c_dim, hidden_size)
            self.fc_v = nn.Linear(c_dim, hidden_size)

            self.fc_di = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            ) # 3d volume
            
            ### multi-head!!!!!
            self.attention = SubAttention(hidden_size=hidden_size//num_heads, num_heads=num_heads)
            c_dim = hidden_size

            if c_dim != 0:
                self.fc_c = nn.ModuleList([
                    nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
                ])
        ###########################################################################
        else:
            self.fc_q1 = nn.Linear(c_dim, hidden_size)
            self.fc_k1 = nn.Linear(c_dim, hidden_size)
            self.fc_v1 = nn.Linear(c_dim, hidden_size)

            self.fc_q2 = nn.Linear(c_dim, hidden_size)
            # self.fc_k2 = nn.Linear(c_dim+2, hidden_size)
            # self.fc_v2 = nn.Linear(c_dim+2, hidden_size)
            self.fc_k2 = nn.Linear(c_dim, hidden_size)
            self.fc_v2 = nn.Linear(c_dim, hidden_size)

            self.fc_q3 = nn.Linear(c_dim, hidden_size)
            # self.fc_k3 = nn.Linear(c_dim+2, hidden_size)
            # self.fc_v3 = nn.Linear(c_dim+2, hidden_size)
            self.fc_k3 = nn.Linear(c_dim, hidden_size)
            self.fc_v3 = nn.Linear(c_dim, hidden_size)

            # learnable positional encoding: displacement vector -> positional encoding

            self.fc_di1 = nn.Sequential(
                nn.Linear(2, c_dim),
                nn.ReLU(),
                nn.Linear(c_dim, c_dim)
            ) # plane 1

            self.fc_di2 = nn.Sequential(
                nn.Linear(2, c_dim),
                nn.ReLU(),
                nn.Linear(c_dim, c_dim)
            ) # plane 2

            self.fc_di3 = nn.Sequential(
                nn.Linear(2, c_dim),
                nn.ReLU(),
                nn.Linear(c_dim, c_dim)
            ) # plane 3

            self.attention1 = SubAttention(hidden_size=hidden_size, num_heads=1)
            self.attention2 = SubAttention(hidden_size=hidden_size, num_heads=1)
            self.attention3 = SubAttention(hidden_size=hidden_size, num_heads=1)

            if c_dim != 0:
                self.fc_c = nn.ModuleList([
                    nn.Linear(c_dim*3, hidden_size) for i in range(n_blocks)
                ])




        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_interpolated_fea = nn.Linear(c_dim, hidden_size)
        # self.fc_af = nn.Linear(c_dim, hidden_size)

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane='xz'):
        # print('pppppp', p.shape)
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        # print('xy', xy.shape)
        xy = xy[:, :, None].float()
        # print('xy', xy.shape)
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        # print('c====', c.shape)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(
            -1).squeeze(-1)
        return c

    def get_plane_feature(self, feature_plane, feature_dim, num_p, idx1, idx2):
        feature_plane = feature_plane.contiguous()
        bs = idx1.shape[0]
        lin_idx = idx2.view(bs,feature_dim,num_p) + feature_plane.size(-1)*idx1.view(bs,feature_dim,num_p)
        feature_plane = feature_plane.view(bs,feature_dim,feature_plane.size(2)*feature_plane.size(3))

        return feature_plane.gather(-1, lin_idx)

    def get_grid_feature(self, feature_volume, feature_dim, num_p, idx1, idx2, idx3):
        feature_volume = feature_volume.contiguous()
        bs = idx1.shape[0]

        lin_idx = idx3.view(bs, feature_dim, num_p) + feature_volume.size(-1) * idx2.view(bs, feature_dim, num_p) + \
                  feature_volume.size(-1) * feature_volume.size(-2) * idx1.view(bs, feature_dim, num_p)
        feature_volume = feature_volume.view(bs, feature_dim, feature_volume.size(2) * feature_volume.size(3) * feature_volume.size(4))

        return feature_volume.gather(-1, lin_idx)

    def find_neighbors_2d(self, plane_size, idx1, idx2):
        bs = idx1.shape[0]
        num_p = idx1.shape[1]
        # left
        x_l = idx1 - 1
        x_l[x_l < 0] = 0
        x_l[x_l > (plane_size - 1)] = plane_size - 1
        # right
        x_r = idx1 + 1
        x_r[x_r < 0] = 0
        x_r[x_r > (plane_size - 1)] = plane_size - 1

        x_stacked = torch.stack([x_l, idx1, x_r, x_l, idx1, x_r, x_l, idx1, x_r], dim=2)
        x_flat = torch.flatten(x_stacked, start_dim=1, end_dim=2)
        x_neighbors = x_flat.view(bs,num_p,9)

        # up
        y_u = idx2 - 1
        y_u[y_u < 0] = 0
        y_u[y_u > (plane_size - 1)] = plane_size - 1
        # down
        y_d = idx2 + 1
        y_d[y_d < 0] = 0
        y_d[y_d > (plane_size - 1)] = plane_size - 1

        y_stacked = torch.stack([y_u, y_u, y_u, idx2, idx2, idx2, y_d, y_d, y_d], dim=2)
        y_flat = torch.flatten(y_stacked, start_dim=1, end_dim=2)
        y_neighbors = y_flat.view(bs,num_p,9)

        return x_neighbors, y_neighbors

    def find_neighbors_3d(self, volume_size, idx1, idx2, idx3):
        bs = idx1.shape[0]
        num_p = idx1.shape[1]
        # left
        x_l = idx1 - 1
        x_l[x_l < 0] = 0
        x_l[x_l > (volume_size - 1)] = volume_size - 1
        # right
        x_r = idx1 + 1
        x_r[x_r < 0] = 0
        x_r[x_r > (volume_size - 1)] = volume_size - 1

        x_stacked = torch.stack([x_l, idx1, x_r, x_l, idx1, x_r, x_l, idx1, x_r, \
                                 x_l, idx1, x_r, x_l, idx1, x_r, x_l, idx1, x_r, \
                                 x_l, idx1, x_r, x_l, idx1, x_r, x_l, idx1, x_r], dim=2)
        x_flat = torch.flatten(x_stacked, start_dim=1, end_dim=2)
        x_neighbors = x_flat.view(bs,num_p,27)

        # up
        y_u = idx2 - 1
        y_u[y_u < 0] = 0
        y_u[y_u > (volume_size - 1)] = volume_size - 1
        # down
        y_d = idx2 + 1
        y_d[y_d < 0] = 0
        y_d[y_d > (volume_size - 1)] = volume_size - 1

        y_stacked = torch.stack([y_u, y_u, y_u, idx2, idx2, idx2, y_d, y_d, y_d, \
                                 y_u, y_u, y_u, idx2, idx2, idx2, y_d, y_d, y_d, \
                                 y_u, y_u, y_u, idx2, idx2, idx2, y_d, y_d, y_d], dim=2)
        y_flat = torch.flatten(y_stacked, start_dim=1, end_dim=2)
        y_neighbors = y_flat.view(bs,num_p,27)

        # front
        z_f = idx3 - 1
        z_f[z_f < 0] = 0
        z_f[z_f > (volume_size - 1)] = volume_size - 1
        # back
        z_b = idx3 + 1
        z_b[z_b < 0] = 0
        z_b[z_b > (volume_size - 1)] = volume_size - 1

        z_stacked = torch.stack([z_f, z_f, z_f, z_f, z_f, z_f, z_f, z_f, z_f, \
                                 idx3, idx3, idx3, idx3, idx3, idx3, idx3, idx3, idx3, \
                                 z_b, z_b, z_b, z_b, z_b, z_b, z_b, z_b, z_b], dim=2)
        z_flat = torch.flatten(z_stacked, start_dim=1, end_dim=2)
        z_neighbors = z_flat.view(bs,num_p,27)

        return x_neighbors, y_neighbors, z_neighbors

    def get_neighbor_disp_feature_2d(self, p, c_plane, plane_flag='xz'):
        bs, num_p, _ = p.shape
        _, feature_dim, plane_size, plane_size = c_plane[plane_flag].shape

        nor_coor = normalize_coordinate(p.clone(), plane=plane_flag, padding=0.1)  # [-0.55, 0.55]->[0,1]
        plane_coor = nor_coor * (plane_size - 1)
        idx = torch.round(plane_coor).long()  # (bs,2048or256,2)
        idx1 = idx[:, :, 0]  # x coordinate
        idx2 = idx[:, :, 1]  # y coordinate
        x_neighbors, y_neighbors = self.find_neighbors_2d(plane_size, idx1, idx2)  # (bs, num_p, 9)

        # get distance
        x_diff = plane_coor[:, :, 0][..., None] - x_neighbors # (bs, num_p, 9)
        y_diff = plane_coor[:, :, 1][..., None] - y_neighbors # (bs, num_p, 9)
        # d_p = torch.sqrt(x_diff ** 2 + y_diff ** 2) #(bs,256,9)
        # d_i = torch.stack((x_diff, y_diff), dim=2).permute(0,3,1,2)  # (bs, 9, num_p, 2)
        d_i = torch.stack((x_diff, y_diff), dim=2).permute(0,1,3,2)  # (bs, num_p, 9, 2)

        x_neighbors_expd = x_neighbors.view(bs, -1)[:, None, :].expand(bs, feature_dim, -1)  # (bs, 32, 256*9)
        y_neighbors_expd = y_neighbors.view(bs, -1)[:, None, :].expand(bs, feature_dim, -1)  # (bs, 32, 256*9)

        neighbor_feature = self.get_plane_feature(c_plane[plane_flag], feature_dim, num_p * 9, x_neighbors_expd,
                                                  y_neighbors_expd)
        # print('neighbor_feature.shape', neighbor_feature.shape) #(bs,32,256*9)
        # neighbor_feature = neighbor_feature.permute(0, 2, 1).view(bs, 9, num_p, feature_dim)  # (bs,9,256,32)
        neighbor_feature = neighbor_feature.permute(0, 2, 1).view(bs, num_p, 9, feature_dim)  # (bs,256,9,32)

        # d_w = 1 - d_p/(np.sqrt(2)*3/2)
        # d_w = d_w[...,None]

        '''
        displacement vector:  d_i (bs,num_p,9,2)
        distance weight: d_w (bs,num_p,9,1)
        neighbor feature:  neighbor_feature (bs,num_p,9,32)
        '''
        return d_i, neighbor_feature

    def get_neighbor_disp_feature_3d(self, p, c_plane, plane_flag='grid'):
        bs, num_p, _ = p.shape
        _, feature_dim, volume_size, volume_size, volume_size = c_plane[plane_flag].shape

        nor_coor = normalize_3d_coordinate(p.clone(), padding=0.1)
        volume_coor = nor_coor * (volume_size - 1) # (bs,256,3)

        idx = torch.round(volume_coor).long()  # (bs,256,3)
        idx1 = idx[:, :, 0]  # x coordinate
        idx2 = idx[:, :, 1]  # y coordinate
        idx3 = idx[:, :, 2]  # z coordinate

        x_neighbors, y_neighbors, z_neighbors = self.find_neighbors_3d(volume_size, idx1, idx2, idx3)  # (bs, num_p, 27)

        # get distance
        x_diff = volume_coor[:, :, 0][..., None] - x_neighbors # (bs, num_p, 27)
        y_diff = volume_coor[:, :, 1][..., None] - y_neighbors # (bs, num_p, 27)
        z_diff = volume_coor[:, :, 2][..., None] - z_neighbors # (bs, num_p, 27)
        # d_p = torch.sqrt(x_diff ** 2 + y_diff ** 2) #(bs,256,9)
        # d_i = torch.stack((x_diff, y_diff), dim=2).permute(0,3,1,2)  # (bs, 9, num_p, 2)
        d_i = torch.stack((x_diff, y_diff, z_diff), dim=2).permute(0,1,3,2)  # (bs, num_p, 27, 3)

        x_neighbors_expd = x_neighbors.view(bs, -1)[:, None, :].expand(bs, feature_dim, -1)  # (bs, 32, 256*27)
        y_neighbors_expd = y_neighbors.view(bs, -1)[:, None, :].expand(bs, feature_dim, -1)  # (bs, 32, 256*27)
        z_neighbors_expd = z_neighbors.view(bs, -1)[:, None, :].expand(bs, feature_dim, -1)  # (bs, 32, 256*27)

        neighbor_feature = self.get_grid_feature(c_plane[plane_flag], feature_dim, num_p * 27, x_neighbors_expd,
                                                  y_neighbors_expd, z_neighbors_expd)
        # print('neighbor_feature.shape', neighbor_feature.shape) #(bs,32,256*27)
        neighbor_feature = neighbor_feature.permute(0, 2, 1).view(bs, num_p, 27, feature_dim)  # (bs,256,27,32)

        # d_w = 1 - d_p/(np.sqrt(2)*3/2)
        # d_w = d_w[...,None]

        '''
        displacement vector:  d_i (bs,num_p,27,3)
        # distance weight: d_w (bs,num_p,9,1)
        neighbor feature:  neighbor_feature (bs,num_p,27,32)
        '''
        return d_i, neighbor_feature


    # def attention(self, q, k, v):
    #     '''
    #     q: (bs, num_p, 1, fea_dim)
    #     k: (bs, num_p, 9, fea_dim)
    #     v: (bs, num_p, 9, fea_dim)
    #     '''
    #     d = q.shape[-1]
    #     scores = torch.einsum('ijkl,ijlm->ijkm', q, k.transpose(2, 3)) / math.sqrt(d)
    #     attention_weights = nn.functional.softmax(scores, dim=-1)
    #     output = torch.einsum('ijkm,ijmn->ijkn', attention_weights, v)
    #     return output


    def forward(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            #print('===0p:', p.shape) #(bs, num_p, 3)
            #print('======c_plane xz shape:', c_plane['xz'].shape) #(bs,32,128,128)

            if 'grid' in plane_type:
                c += self.sample_grid_feature(p, c_plane['grid'])
                c = c.transpose(1,2)
            if 'xz' in plane_type:
                c_xz = self.sample_plane_feature(p, c_plane['xz'], plane='xz')
                c_xz = c_xz.transpose(1, 2)
            if 'xy' in plane_type:
                c_xy = self.sample_plane_feature(p, c_plane['xy'], plane='xy')
                c_xy = c_xy.transpose(1, 2)
            if 'yz' in plane_type:
                c_yz = self.sample_plane_feature(p, c_plane['yz'], plane='yz')
                c_yz = c_yz.transpose(1, 2)
            # c = c.transpose(1, 2)
            # c = c_xz + c_xy + c_yz


        ### transformer
        p = p.float() # (bs, 2048or56 ,3)
        #inputs = inputs.float() #(bs,10000,3)

        bs, num_p, _ = p.shape
        # _, feature_dim, plane_size, plane_size = c_plane['xz'].shape

        if 'grid' in plane_type:
            d_i, neighbor_feature = self.get_neighbor_disp_feature_3d(p, c_plane, plane_flag='grid')
            NF = neighbor_feature
            pos_encoding = self.fc_di(d_i)
        if 'xz' in plane_type:
            d_i_xz, neighbor_feature_xz = self.get_neighbor_disp_feature_2d(p, c_plane, plane_flag='xz')
            NF_xz = neighbor_feature_xz
            pos_encoding_xz = self.fc_di1(d_i_xz)
        if 'xy' in plane_type:
            d_i_xy, neighbor_feature_xy = self.get_neighbor_disp_feature_2d(p, c_plane, plane_flag='xy')
            # NF_xy = torch.cat((neighbor_feature_xy, d_i_xy), dim=3)
            NF_xy = neighbor_feature_xy
            pos_encoding_xy = self.fc_di2(d_i_xy)
        if 'yz' in plane_type:
            d_i_yz, neighbor_feature_yz = self.get_neighbor_disp_feature_2d(p, c_plane, plane_flag='yz')
            # NF_yz = torch.cat((neighbor_feature_yz, d_i_yz), dim=3)
            NF_yz = neighbor_feature_yz
            pos_encoding_yz = self.fc_di3(d_i_yz)

        '''
        query pts interpolated feature c: (bs,256,32) x 3
        neighbor_feature NF: (bs,9,256,32) x 3
        '''

        # Attention is all you need!
        if 'grid' in plane_type:
            q = self.fc_q(c)[:,:,None,:] # (32,256,32)->(32,256,1,32)
            k = self.fc_k(NF) # (bs,256,27,32)
            v = self.fc_v(NF) # (bs,256,27,32)
            attention_output = self.attention(q, k, v, pos_encoding)

        if 'xz' in plane_type:
            q1 = self.fc_q1(c_xz)[:,:,None,:] # (32,256,32)->(32,256,1,32)
            k1 = self.fc_k1(NF_xz) # (32,256,9,32)
            v1 = self.fc_v1(NF_xz) # (32,256,9,32)
            attention_output1 = self.attention1(q1, k1, v1, pos_encoding_xz)#.squeeze(2)  # (32,256,1,32)->(32,256,32)

        if 'xy' in plane_type:
            q2 = self.fc_q2(c_xy)[:,:,None,:] # (32,256,32)->(32,256,1,32)
            k2 = self.fc_k2(NF_xy) # (32,256,9,32)
            v2 = self.fc_v2(NF_xy) # (32,256,9,32)
            attention_output2 = self.attention2(q2, k2, v2, pos_encoding_xy)#.squeeze(2)  # (32,256,1,32)->(32,256,32)

        if 'yz' in plane_type:
            q3 = self.fc_q3(c_yz)[:,:,None,:] # (32,256,32)->(32,256,1,32)
            k3 = self.fc_k3(NF_yz) # (32,256,9,32)
            v3 = self.fc_v3(NF_yz) # (32,256,9,32)
            attention_output3 = self.attention3(q3, k3, v3, pos_encoding_yz)#.squeeze(2)  # (32,256,1,32)->(32,256,32)


        if self.plane_type == 'grid':
            attention_fea = attention_output
        else:
            attention_fea = torch.cat((attention_output1, attention_output2, attention_output3), dim=2)

        ### 1. MLP
        # net = self.fc_af(attention_fea) #0.5937
        ### 2. Resnet
        # net = self.fc_interpolated_fea(c)
        net = 0
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](attention_fea)

            net = self.blocks[i](net)


        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        p_r = dist.Bernoulli(logits=out)
        return p_r


############################# ^ Attention decoder ^ #####################################

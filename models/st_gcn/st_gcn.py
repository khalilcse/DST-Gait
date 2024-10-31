import torch
import torch.nn as nn
import torch.nn.functional as F

from models.st_gcn.tgcn import ConvTemporalGraphical


class STGCNEmbedding(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes
    """

    def __init__(self, in_channels, graph, edge_importance_weighting=True, temporal_kernel_size=3, #9
                 embedding_layer_size=128, **kwargs):
        super().__init__()

        # load graph
        self.graph = graph
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        # temporal_kernel_size = 9

        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        
                
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=True, **kwargs0), # original residual=False
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            #st_gcn(64, 64, kernel_size, 1, **kwargs),
            #st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 1, **kwargs), # original fourth parameter 2
            #st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 1, **kwargs), # original fourth parameter 2
            #st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()), requires_grad=True)
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fcn = nn.Linear(256, embedding_layer_size)

    def forward(self, x):
        #print('From forward of ST-GCN')
        x = x.permute(0, 3, 4, 1, 2).squeeze()

        # data normalization
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)

        x = self.data_bn(x)
        x = x.view(N, V, C, T)

        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V) # 256, 3, 60, 17
        # forward
        #print('Start from the ST-GCN function at forward ...')
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        feature = x

        x = self.global_pooling(x)
        x = self.fcn(x.squeeze())

        return x, feature

    # Alias for model.forward()
    def get_embedding(self, x):
        return self.forward(x)

######################## SE Mechanism ###################################
"""
class SqueezeAndExcitation(nn.Module):
    def __init__(self, in_channels, out_channels): #, reduction=16
        super(SqueezeAndExcitation, self).__init__()
        self.sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Squeeze operation using adaptive pooling
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        excitation = self.sequential(x)
        return x * excitation
#---------------------------------------------------------------------------------

# Attention Model with ES-Mechanism with reduction value 8 or 16
class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SqueezeAndExcitation, self).__init__()
        self.sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Squeeze operation using adaptive pooling
            nn.Conv2d(channels, max(channels // reduction, 1), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 1), channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        excitation = self.sequential(x)
        return x * excitation
#-----------------------------------------------------------------------------------------------
"""

# DSE- Mechanism - AVGPool + 1x1 Conv2D + Sigmoid
class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels): #, reduction=16
        super(SqueezeAndExcitation, self).__init__()
        self.sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Squeeze operation using adaptive pooling
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        excitation = self.sequential(x)
        return x * excitation # here x is used as the residual connection
#-----------------------------------------------------------------------------------------------

"""
class SqueezeAndExcitation(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(SqueezeAndExcitation, self).__init__()
        reduced_channels = max(out_channels // reduction, 1)  # Ensure at least 1 channel after reduction
        self.sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Squeeze operation
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, out_channels, kernel_size=1),
            nn.Sigmoid()  # Excitation operation
        )
    def forward(self, x):
        excitation = self.sequential(x)
        return x * excitation
"""
#########################################################################

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        
        #print('Start GCN New')
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        
        #print('Start TCN New')
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            # nn.Dropout(dropout, inplace=True),
        )
       
        
        # *************** Main Code Block *******************************************
        # This block is important for the re-shapping the cannel i.e., channel = 3 to channel = 64 for first ST-GCN
        if not residual:
            #print('Residual No Connection')
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            #print('Residual Single Connection')
            self.residual = lambda x: x

        else:
            #print('Residual with Conv2D Connection - Reshape the channel')
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True) # exist at the main code
        
        #print('Residual Single Connection')
        #self.residual = lambda x: x
        # ********************** Main Code Block End **************************************
        # ************** Residual connection with Avg_pool + 1x1 Conv2d + sigmoid ***********
        
        """
        print('Attention mechanism: Residual with 1x1 Conv2D Connection New')
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        """
        # ***********************************************************************
        #print('Residual with 1x1 Conv2D Connection without reduction')
        #self.residual = SqueezeAndExcitation(in_channels, out_channels)
        #print('Residual with 1x1 Conv2D Connection with reduction')
        self.residual_att = SqueezeAndExcitation(out_channels) # DSE Block
        
    def forward(self, x, A):
        #print(f"Size fo X:{x.shape}")
        res_reshape = self.residual(x) # reshape to 3 to 64 channels
        #print(f"Size fo res_reshape:{res_reshape.shape}")
        #attention_weights = self.attention(x)
        res = self.residual_att(res_reshape) # pass for the attention SE-Attention
        #res = res_reshape
        #print(f"Size fo res+att:{res.shape}")
        x, A = self.gcn(x, A)
        #print(f"Size fo GCN:{x.shape}")
        #x = self.tcn(x) + res
        xt = self.tcn(x)
        #print(f"Size fo TCN:{xt.shape}")
        x  = xt + res 
        return self.relu(x), A

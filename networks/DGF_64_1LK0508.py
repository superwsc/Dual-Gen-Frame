import torch
import sys
sys.path.append("..")
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from networks.ReverseDiffusion import Unet, GaussianDiffusion

#from networks.wavelet import DWT, IWT
from einops import rearrange
from networks.mcc import MCC
from utils.antialias import Edge as edge

from PIL import Image
# from mcc import MCC
# from antialias import Edge as edge
########################################################################## 



## SA CA
class SA_CA(nn.Module):
    def __init__(self, in_channels, height=3,reduction=8,bias=False, kernel_size=5):
        super(SA_CA, self).__init__()
        
        self.height = height

        self.sas = nn.ModuleList([])
        for i in range(self.height):
            self.sas.append(SALayer())
            
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels*3, in_channels, 1, padding=0, bias=bias), nn.PReLU())

        self.CA = CALayer(channel = in_channels*3)

        #self.conv2_1 = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, 1, padding=0, bias=bias), nn.PReLU())
    def forward(self, inp_feats):
       
        inp_feats = torch.cat(inp_feats, dim=1)

        inp_feats_2 = self.conv1_1(inp_feats)

        SA_vectors = [sa(inp_feats_2) for sa in self.sas]

        SA_vectors = torch.cat(SA_vectors, dim=1)


        ##########channel wise
        feats_V = self.CA(SA_vectors)
        out = self.conv1_1(feats_V)

        #print("line115")

        return out      


##########################################################################
# Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # torch.max will output 2 things, and we want the 1st one
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  # [N,2,H,W]  could add 1x1 conv -> [N,3,H,W]
        y = self.conv_du(channel_pool)

        return x * y

##########################################################################
# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



class PLF(nn.Module):
    def __init__(self, n_feat=64, kernel_size=3, reduction=16, bias=False, act=nn.PReLU()):
        super(PLF, self).__init__()

        modules_body = \
        [
        nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
        nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=bias, padding_mode='reflect', groups=n_feat),
        ]

        self.body = nn.Sequential(*modules_body)
        self.act = act

        self.edge_extract = edge(channels=n_feat,filt_size=3,stride=1)

        self.down = nn.Conv2d(n_feat, n_feat * 2 , kernel_size=4, stride=2, padding=1, bias=bias)
        self.up = nn.ConvTranspose2d(n_feat * 2 , n_feat, kernel_size=2, stride=2, bias=True)


        modules_body2 = \
        [
        nn.Conv2d(n_feat * 2 , n_feat * 2, kernel_size=1, bias=bias),
        nn.Conv2d(n_feat * 2 , n_feat * 2 , 3, 1, 1, bias=bias, padding_mode='reflect', groups=n_feat),
        ]

        self.body2 = nn.Sequential(*modules_body2)

        self.edge_extract2 = edge(channels=n_feat * 2 ,filt_size=3,stride=1)

        self.mcc = MCC(f_number = n_feat, num_heads= 8, padding_mode='reflect')

        self.Cross_Attention1 = Cross_Attention(n_feat * 2, n_feat * 2)
        self.Cross_Attention2 = Cross_Attention(n_feat , n_feat )

        self.conv1 = nn.Conv2d(n_feat * 2 , n_feat, kernel_size=1, bias=bias)

        self.learnable_vector1 = nn.Parameter(torch.ones(1, n_feat, 1, 1))
        self.learnable_vector2 = nn.Parameter(torch.ones(1, n_feat*2, 1, 1))


    def forward(self, x):
        spatial1 = self.act(self.body(x))
        edge1 = self.body(self.edge_extract(x) * self.learnable_vector1)

        down1 = self.down(spatial1)
        spatial2 = self.act(self.body2(down1))
        edge2 = self.body2(self.edge_extract2(down1)* self.learnable_vector2)

        
        ssa1 = self.Cross_Attention1(spatial2, edge2)
        ssa2 = self.Cross_Attention2(spatial1, edge1)


        up1 = self.up(ssa1)

        concat = torch.cat([ssa2, up1], 1)
        
        out = self.conv1(concat)

        out = out + x

        return out



class ConvBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope = 0.2):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(out_size, out_size, kernel_size=1, bias=True),
            nn.Conv2d(out_size, out_size, 3, 1, 1, bias=True, padding_mode='reflect', groups=out_size),
            nn.LeakyReLU(relu_slope))

        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc

        return out


class Cross_Attention(nn.Module):

    def __init__(self, in_size, out_size, relu_slope = 0.2 , subspace_dim=16):
        super(Cross_Attention, self).__init__()
        #self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = ConvBlock(in_size, out_size, relu_slope)
        self.num_subspace = subspace_dim
       

        self.temperature = nn.Parameter(torch.ones(subspace_dim, 1, 1))

    def forward(self, x1, x2):
        

        b_, c_, h_, w_ = x2.shape

        q = rearrange(x1, 'b (head c) h w -> b head c (h w)', head=self.num_subspace)
        k = rearrange(x1, 'b (head c) h w -> b head c (h w)', head=self.num_subspace)
        v = rearrange(x2, 'b (head c) h w -> b head c (h w)', head=self.num_subspace)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_subspace, h=h_, w=w_)
        out = self.conv_block(out)

        out = out + x1
        return out
    


class MappingFunction(nn.Module):
    def __init__(self):
        super(MappingFunction, self).__init__()
        self.n = nn.Parameter(torch.rand(1, requires_grad=True))
        self.s = nn.Parameter(torch.rand(1, requires_grad=True))

    def forward(self, input_image):
        x = input_image

        y = 0.257 * x[:, 0, :, :] + 0.564 * x[:, 1, :, :] + 0.098 * x[:, 2, :, :] + 0.0626
        u = -0.148 * x[:, 0, :, :] -0.291 * x[:, 1, :, :] + 0.439 * x[:, 2, :, :] + 0.5001
        v = 0.439 * x[:, 0, :, :] - 0.368 * x[:, 1, :, :] - 0.071 * x[:, 2, :, :] + 0.5001
        y = torch.clamp(y, 0, 1)
        u = torch.clamp(u, 0, 1)
        v = torch.clamp(v, 0, 1)
        
        y = y.unsqueeze(1)
        u = u.unsqueeze(1)
        v = v.unsqueeze(1)

        n =  torch.abs(self.n)
        s =  torch.abs(self.s)
      
        


        Yc = (y**n)/(y**n + s**n)

        
        if torch.isnan(n).any():
            print("n is NAN")
            print(f"n is {n}")
            print(f"s is {s}")

            sys.exit()
        if torch.isnan(s).any():
            print("s is NAN")
            print(f"s is {s}")
            print(f"n is {n}")
            sys.exit()
        

        r1 = 1.164 *(Yc - 0.0626)  + 1.596 * (v - 0.5001)
        g1 = 1.164 *(Yc - 0.0626)  - 0.392 * (u - 0.5001) - 0.813 * (v - 0.5001)
        b1 = 1.164 *(Yc - 0.0626)  + 2.017 * (u - 0.5001)
        
        rgb1 = torch.cat([r1,g1,b1],dim = 1)

        rgb1 = torch.clamp(rgb1, 0, 1)
        
        return rgb1, self.n, self.s
    
class MappingFunctionConcat(nn.Module):

    count = 0

    def __init__(self,in_channels=9, n_feat=64, bias = False):
        super(MappingFunctionConcat, self).__init__()
        self.mapping_function1 = MappingFunction()
        self.mapping_function2 = MappingFunction()
        self.mapping_function3 = MappingFunction()
        # 其他模型组件...
        self.body = nn.Sequential(nn.Conv2d(in_channels, n_feat, 1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1, bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(n_feat, n_feat, 1, stride=1, padding=0, bias=bias))
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(64, 6, 1, stride=1, padding=0, bias=bias))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(64, 6, 1, stride=1, padding=0, bias=bias))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                nn.Conv2d(64, 64, 3, 1, 1, bias=True, padding_mode='reflect', groups=64),
                                nn.PReLU(),
                                nn.Conv2d(64, 6, 1, stride=1, padding=0, bias=bias))
        


    def forward(self, input_image , input_feature):
        mapped1, n1, s1 = self.mapping_function1(input_image)
        mapped2, n2, s2 = self.mapping_function2(input_image)
        mapped3, n3, s3 = self.mapping_function3(input_image)


        feature1 = self.conv1(input_feature)
        feature2 = self.conv2(input_feature)
        feature3 = self.conv3(input_feature)
        A1, B1 = torch.chunk(feature1, 2, dim=1)
        A2, B2 = torch.chunk(feature2, 2, dim=1)
        A3, B3 = torch.chunk(feature3, 2, dim=1)
        mapped1 = A1 * mapped1 + B1
        mapped2 = A2 * mapped2 + B2
        mapped3 = A3 * mapped3 + B3
        mapped1 = torch.clamp(mapped1 , 0, 1)
        mapped2 = torch.clamp(mapped2 , 0, 1)
        mapped3 = torch.clamp(mapped3 , 0, 1)




        MappingFunctionConcat.count = MappingFunctionConcat.count + 1

        mapped = torch.cat([mapped1, mapped2, mapped3], dim = 1)
        out = self.body(mapped)

        return out
    

class LongRange(nn.Module):
    def __init__(self, f_number, num_heads=2, padding_mode='reflect', bias=False) -> None:
        super().__init__()
        

        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #point wise
        self.pwconv = nn.Conv2d(f_number, f_number * 3, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(f_number, f_number, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape
        #qkv = self.dwconv(self.pwconv(attn))
        qkv = self.pwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        #out = self.feedforward(out + x)
        return out

##########################################################################

class FFB(nn.Module):
    def __init__(self, n_feat, height, width, stride, bias):
        super(FFB, self).__init__()

        self.n_feat, self.height, self.width = n_feat, height, width

        #[B C H W] = [B 3 H W]
        self.map = MappingFunctionConcat()
        
        self.plf = PLF()
        self.conv1_1 = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, padding=0, bias=bias)
        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=bias)

        self.dilated_conv = nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, bias=bias)
        self.dilated_conv3 = nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, dilation=3,padding=3, bias=bias)

        self.long_range = LongRange(f_number=n_feat)

        self.selective_kernel = SA_CA(n_feat, height) 

    def forward(self, input):
        x = input[0]
        input_image = input[1]

        mapped = self.map(input_image, x)
        inp = torch.cat([x , mapped],1)
        inp2 = self.conv1_1(inp)

        plf1 = self.plf(inp2)


        dilated1 = self.dilated_conv(plf1)

        dilated3 = self.dilated_conv3(plf1)
        long_range = self.long_range(plf1)


        sa_ca = self.selective_kernel([dilated1,dilated3,long_range])

        plf2 = self.plf(sa_ca)
        
        out = self.conv_out(plf2)
        out = out + x

        return [out, input_image]

    def select_up_down(self, tensor, j, k):
        if j==k:
            return tensor
        else:
            diff = 2 ** np.abs(j-k)
            if j<k:
                return self.up[f'{tensor.size(1)}_{diff}'](tensor)
            else:
                return self.down[f'{tensor.size(1)}_{diff}'](tensor)


    def select_last_up(self, tensor, k):
        if k==0:
            return tensor
        else:
            return self.last_up[f'{k}'](tensor)


##########################################################################
class FFB_wrapper(nn.Module):
    def __init__(self, n_feat, n_FFB, height, width, stride, bias=False):
        super(FFB_wrapper, self).__init__()

        modules_body = [FFB(n_feat, height, width, stride, bias) for _ in range(n_FFB)]
        self.conv = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=bias)

        self.body = nn.Sequential(*modules_body)

    def forward(self, input):
        x = input[0]
        input_image = input[1]
        [res, _ ]= self.body(input)
        res = self.conv(res)
        res += x

        input[0] = res
        return input

class DGF(nn.Module):
    def __init__(self,device, in_channels=3, out_channels=3, n_feat=64, kernel_size=3, stride=2, n_FFB_wrapper=3, n_FFB=2, height=3, width=2, bias=False):
        super(DGF, self).__init__()
        self.device = device
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias)

        modules_body = [FFB_wrapper(n_feat, n_FFB, height, width, stride, bias) for _ in range(n_FFB_wrapper)]

        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias)

        self.mcc = MCC(f_number = n_feat, num_heads= 8, padding_mode='reflect')
        
    def forward(self, x):
        h = self.conv_in(x)
        out_fea1 = h[:,:16,:,:]
        [h, _] = self.body([h,x])


        out_fea2 = h[:,:16,:,:]
        h = self.conv_out(h)

        h += x

        out_fea = torch.cat([out_fea1,out_fea2],1)
        return h, out_fea


if __name__ == "__main__":
    from thop import profile
    torch.backends.cudnn.enabled = False
    input = torch.ones(1, 3, 256, 256, dtype=torch.float, requires_grad=False).cuda()

    model = DGF(device = 'cuda:3').cuda()


    out = model(input)
    flops, params = profile(model, inputs=(input,))

    print('input shape:', input.shape)
    print('parameters:', params/1e6, 'M')
    print('flops', flops/1e9 , 'G')
    print('output shape', out[0].shape)

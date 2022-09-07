from vit_pytorch.vit import ViT
from vit_pytorch.recorder import Recorder
import torch
import torch.nn as nn

from cvt import CvT

from prettytable import PrettyTable

num_heads = 8
dep = 5

class ViT_density(nn.Module):
    def __init__(self):
        super(ViT_density, self).__init__()
        self.cvt = CvT(s1_emb_dim = 64,
        s1_emb_kernel = 7,
        s1_emb_stride = 4,
        s1_proj_kernel = 3,
        s1_kv_proj_stride = 2,
        s1_heads = 1,
        s1_depth = 1,
        s1_mlp_mult = 4,
        s2_emb_dim = 64,
        s2_emb_kernel = 3,
        s2_emb_stride = 2,
        s2_proj_kernel = 3,
        s2_kv_proj_stride = 2,
        s2_heads = 3,
        s2_depth = 5,#2
        s2_mlp_mult = 4,
        s3_emb_dim = 64,#384
        s3_emb_kernel = 3,
        s3_emb_stride = 1,
        s3_proj_kernel = 3,
        s3_kv_proj_stride = 2,
        s3_heads = 6,
        s3_depth = 5,
        s3_mlp_mult = 4,
        dropout = 0.1)
        self.bn=nn.BatchNorm2d(64)

        #self.conv_1 = nn.Conv2d(64, 1, kernel_size=1,padding='same')
        # self.conv_1 = nn.Sequential(
        #     nn.Conv2d(num_heads*dep, 256, (6, 6), stride=2),
        #     nn.Conv2d(256, 128, (9, 9), stride=1),
        #     #nn.Conv2d(128, 128, (3, 3), padding='same', dilation=2),
        #     nn.Conv2d(128, 64, (9, 9)),
        #     nn.Conv2d(64, 64, (9, 9)),
        #     nn.Conv2d(64, 32, (9, 9)),
        #     nn.Conv2d(32, 32, (9, 9)),
        #     nn.Conv2d(32, 32, (9, 9)),
        #     nn.Conv2d(32, 32, (9, 9)),
        #     nn.Conv2d(32, 32, (7, 7))
        # )

        # # Conv layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding='same', dilation=2),
            nn.Conv2d(64, 64, (3, 3), padding='same', dilation=2),
            #nn.Conv2d(128, 128, (3, 3), padding='same', dilation=2),
            nn.Conv2d(64, 64, (3, 3), padding='same', dilation=2),
            nn.Conv2d(64, 64, (3, 3), padding='same', dilation=2),
            nn.Conv2d(64, 32, (3, 3), padding='same', dilation=2),
            nn.Conv2d(32, 1, (3, 3), padding='same', dilation=1)
        )

        # deconvolution layers
        # self.conv_block = nn.Sequential(
        #     nn.ConvTranspose2d(192, 32, kernel_size=6, stride=1),
        #     nn.ConvTranspose2d(32, 32, kernel_size=6, stride=1),
        #     nn.ConvTranspose2d(32, 32, kernel_size=6, stride=1),
        #     nn.ConvTranspose2d(32, 32, kernel_size=6, stride=1),
        #     nn.ConvTranspose2d(32, 32, kernel_size=6, stride=1),
        #     nn.ConvTranspose2d(32, 32, kernel_size=6, stride=1),
        #     nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1),
        #     nn.ConvTranspose2d(32, 1, kernel_size=1, stride=1)
        # )

    def forward(self, x):
        batch_size = x.shape[0]
        #x = self.vit(x)
        x = x.float()
        x = self.cvt(x)

        #x= self.bn(x) #loss 195, MAE train: 850, last MAE:15000, best MAE:336
        
        #x = self.conv_1(x) #loss 125, MAE train: 71, last MAE:26000, best MAE:446, without bn
        x = self.conv_block(x) ##loss 147, MAE train: 50, last MAE:5000, best MAE:448, without bn and conv1
        #print(x.shape, type(x)) 
        
        return x

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

if __name__ == '__main__':
    model = ViT_density()
    img = torch.randn(1,3, 512, 512)
    x = model(img)
    print(x.shape)

    count_parameters(model)



# class ViT_density(nn.Module):
#     def __init__(self):
#         super(ViT_density, self).__init__()
#         self.vit = ViT(
#                     image_size = 512,
#                     patch_size = 32,
#                     num_classes = 1,
#                     dim = 128,#1024
#                     depth = dep,#6
#                     heads = num_heads,#8
#                     mlp_dim = 128,#1024
#                     dropout = 0.1,
#                     emb_dropout = 0.1
#                 )
#         self.recorder = Recorder(self.vit)
#         #self.bn=nn.BatchNorm2d(num_heads*dep)

#         #self.conv_1 = nn.Conv2d(num_heads*dep, 32, kernel_size=2, stride=1)
#         # self.conv_1 = nn.Sequential(
#         #     nn.Conv2d(num_heads*dep, 256, (6, 6), stride=2),
#         #     nn.Conv2d(256, 128, (9, 9), stride=1),
#         #     #nn.Conv2d(128, 128, (3, 3), padding='same', dilation=2),
#         #     nn.Conv2d(128, 64, (9, 9)),
#         #     nn.Conv2d(64, 64, (9, 9)),
#         #     nn.Conv2d(64, 32, (9, 9)),
#         #     nn.Conv2d(32, 32, (9, 9)),
#         #     nn.Conv2d(32, 32, (9, 9)),
#         #     nn.Conv2d(32, 32, (9, 9)),
#         #     nn.Conv2d(32, 32, (7, 7))
#         # )

#         # # # Conv layers
#         # self.conv_block = nn.Sequential(
#         #     nn.Conv2d(32, 64, (3, 3), padding='same', dilation=2),
#         #     nn.Conv2d(64, 64, (3, 3), padding='same', dilation=2),
#         #     #nn.Conv2d(128, 128, (3, 3), padding='same', dilation=2),
#         #     nn.Conv2d(64, 64, (3, 3), padding='same', dilation=2),
#         #     nn.Conv2d(64, 64, (3, 3), padding='same', dilation=2),
#         #     nn.Conv2d(64, 32, (3, 3), padding='same', dilation=2),
#         #     nn.Conv2d(32, 1, (3, 3), padding='same', dilation=1)
#         # )

#         # deconvolution layers
#         self.conv_block = nn.Sequential(
#             nn.ConvTranspose2d(257*num_heads//2, 32, kernel_size=9, stride=1),
#             nn.ConvTranspose2d(32, 32, kernel_size=6, stride=2),
#             nn.ConvTranspose2d(32, 128, kernel_size=5, stride=1),
#             nn.ConvTranspose2d(128, 128, kernel_size=5, stride=1),
#             nn.ConvTranspose2d(128, 32, kernel_size=5, stride=1),
#             nn.ConvTranspose2d(32, 1, kernel_size=1, stride=1)
#         )

#     def forward(self, x, train=True):
#         batch_size = x.shape[0]
#         #x = self.vit(x)
#         x = x.float()
#         self.preds, self.attns = self.recorder(x)

#         #print(self.attns.shape)

#         #attn_heatmap = attns[:,-1,-1, :, 1:].reshape((batch_size,257,16, 16))#.detach().cpu().numpy()
#         #[batch size, depth, heads, rows, w, h]
#         attn_heatmap = self.attns[:,-1,::2,:,1:].reshape((batch_size,257*num_heads//2,16,16))

#         #1, 3, 6, 257, 257
#         #attn_heatmap = self.attns.reshape((batch_size,num_heads*dep,self.attns.shape[-1],self.attns.shape[-1]))
#         #print(attn_heatmap.shape)

#         #x= self.bn(attn_heatmap)
        
#         #x = self.conv_1(attn_heatmap)
#         #print(attn_heatmap.shape, 257*num_heads//2)
#         x = self.conv_block(attn_heatmap)
#         #print(x.shape, type(x))

#         if not train:
#           return x, self.attns
        
#         return x
"""U-Net adapted to exactly the colleague dataset's S/Q/P information."""
import torch
from torch import nn
from torch.nn import functional as F
from baseline_model import Block


class MatchedEncoder(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.blocks=nn.ModuleList([Block(3,c),Block(c,2*c),Block(2*c,4*c),Block(4*c,8*c)])

    def forward(self, spatial, side=None):
        features=[]
        for i, block in enumerate(self.blocks):
            if i: spatial=F.max_pool2d(spatial,2)
            spatial=block(spatial)
            if i==0 and side is not None: spatial=spatial+side
            features.append(spatial)
        return features


class MatchedCNN(nn.Module):
    def __init__(self, source=False):
        super().__init__()
        c=16
        self.encoder=MatchedEncoder(c)
        self.decoder=nn.ModuleList([Block(12*c,4*c),Block(6*c,2*c),Block(3*c,c)])
        self.head=nn.Conv2d(c,1,1)
        self.source=source
        if not source:
            self.side_embed=nn.Conv2d(3,c,1)
            nn.init.zeros_(self.side_embed.weight); nn.init.zeros_(self.side_embed.bias)
            self.film=nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,16*c))
            nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)

    def forward(self, batch):
        spatial=batch.spatial
        h,w=spatial.shape[-2:]
        pad=(0,(-w)%8,0,(-h)%8)
        side=None if self.source else self.side_embed(F.pad(batch.pattern_maps,pad))
        features=self.encoder(F.pad(spatial,pad),side)
        x=features[-1]
        if not self.source:
            gamma,beta=self.film(batch.numeric_condition).chunk(2,dim=1)
            x=x*(1+gamma[:,:,None,None])+beta[:,:,None,None]
        for block,skip in zip(self.decoder,reversed(features[:-1])):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=block(torch.cat((x,skip),dim=1))
        return self.head(x)[:,:,:h,:w]

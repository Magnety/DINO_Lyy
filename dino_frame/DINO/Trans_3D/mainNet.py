import torch


#from CoTr.network_architecture import CNNBackbone
#from CoTr.network_architecture.neural_network import SegmentationNetwork
def posi_mask(self, x):
    x_fea = []
    x_posemb = []
    masks = []
    for lvl, fea in enumerate(x):
        if lvl > 1:
            x_fea.append(fea)
            x_posemb.append(self.position_embed(fea))
            masks.append(torch.zeros((fea.shape[0], fea.shape[2], fea.shape[3], fea.shape[4]), dtype=torch.bool).cuda())
    return x_fea, masks, x_posemb

from networks.unet2d import UNet2D

def net_factory(net_type="unet"):
    if net_type == "unet":
        net = UNet2D().cuda()
    else:
        net = None 
    return net
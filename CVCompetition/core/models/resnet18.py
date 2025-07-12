import torch
import torch.nn as nn
import timm

class BinaryResnet18(nn.Module):
    def __init__(self, cfg):
        super(BinaryResnet18, self).__init__()
        print("create model binary resnet18")

        self.cfg = cfg

        if cfg.loss.loss_name.lower() == "bceloss":
            self.num_classes = 1
        elif cfg.loss.loss_name.lower() == "crossentropy":
            self.num_classes = 2
        else:
            raise ValueError(f"[ERROR] unknown loss: {cfg.loss.loss_name}")
        
        self.backbone = timm.create_model(cfg.model.model.model_name,
                                          pretrained=cfg.model.model.pretrained,
                                          num_classes=self.num_classes)
        
        for name, param in self.backbone.named_parameters():
            if 'fc' in name or 'head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.loss_name = self.cfg.loss.loss_name.lower()

    def forward(self, x):
        out = self.backbone(x)
        if self.cfg.loss.loss_name.lower() == "bceloss":
            return out.squeeze(1)
        return out # (B, 2)
    
    def unfreeze_mid(self):
        for name, param in self.backbone.named_parameters():
            if any(k in name for k in ["layer4", "fc", "head"]):  # fc: classifier
                param.requires_grad = True

    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

class Resnet18(nn.Module):
    def __init__(self, cfg):
        super(Resnet18, self).__init__()
        print("create model resnet18")
        self.backbone = timm.create_model(cfg.model.model.model_name,
                                          pretrained=cfg.model.model.pretrained,
                                          num_classes=cfg.model.model.num_classes)
        
        for name, param in self.backbone.named_parameters():
            if 'fc' in name or 'head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        return x
    
    def unfreeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = True
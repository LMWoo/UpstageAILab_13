import torch
import torch.nn as nn
import timm

class ViT(nn.Module):
    def __init__(self, cfg):
        super(ViT, self).__init__()
        print("create model ViT")
        self.backbone = timm.create_model(cfg.model.model.model_name,
                                          pretrained=cfg.model.model.pretrained,
                                          num_classes=cfg.model.model.num_classes)
        
        for name, param in self.backbone.named_parameters():
            if 'head' in name or 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        return x
    
    def unfreeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = True

class BinaryViT(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        print("create model binary ViT")

        loss_name = cfg.loss.loss_name.lower()
        if loss_name == "bceloss":
            num_classes = 1
        elif loss_name == "crossentropyloss":
            num_classes = 2
        else:
            raise ValueError(f"[ERROR] unknown loss: {cfg.loss.loss_name}")

        self.backbone = timm.create_model(
            cfg.model.model.model_name,
            pretrained=cfg.model.model.pretrained,
            num_classes=num_classes,
        )

        for name, p in self.backbone.named_parameters():
            p.requires_grad = "head" in name or "classifier" in name

        self.loss_name = loss_name

    # --------------------------------------------------
    def forward(self, x):
        out = self.backbone(x)           # (B, 1) or (B, 2)
        if self.loss_name == "bceloss":
            return out.squeeze(1)        # (B,)
        return out

    # --------------------------------------------------
    def unfreeze_mid(self):
        for name, p in self.backbone.named_parameters():
            if any(name.startswith(f"blocks.{i}") for i in [9, 10, 11]) or \
               "head" in name or "classifier" in name:
                p.requires_grad = True

    # --------------------------------------------------
    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
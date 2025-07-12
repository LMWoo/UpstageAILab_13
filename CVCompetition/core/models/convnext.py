import torch
import torch.nn as nn
import timm

class ConvNeXt(nn.Module):
    def __init__(self, cfg):
        super(ConvNeXt, self).__init__()
        print("create model convnext")
        self.backbone = timm.create_model(cfg.model.model.model_name,
                                          pretrained=cfg.model.model.pretrained,
                                          num_classes=cfg.model.model.num_classes)
        
        # 초기에는 head만 학습
        self.freeze_all()
        self.unfreeze_head()

    def forward(self, x):
        x = self.backbone(x)
        return x
    
    def freeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_head(self):
        for name, param in self.backbone.named_parameters():
            if "head" in name or "classifier" in name:
                param.requires_grad = True

    def unfreeze_mid(self):
        for name, param in self.backbone.named_parameters():
            if any([k in name for k in ["stages.3", "stages.2", "norm", "head", "classifier"]]):
                param.requires_grad = True

    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

class BinaryConvNeXt(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print("create model binary convnext")

        self.cfg = cfg
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

        # 기본: head만 학습
        for name, p in self.backbone.named_parameters():
            p.requires_grad = "head" in name

        self.loss_name = loss_name

    # --------------------------------------------------
    def forward(self, x):
        out = self.backbone(x)
        if self.loss_name == "bceloss":
            return out.squeeze(1)
        return out

    # --------------------------------------------------
    def unfreeze_mid(self):
        for name, p in self.backbone.named_parameters():
            if any(s in name for s in ["stages.3", "stages.2", "head"]):
                p.requires_grad = True

    # --------------------------------------------------
    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
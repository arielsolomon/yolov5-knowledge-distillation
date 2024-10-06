import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import Conv

class FeatureBasedDistillation(nn.Module):
    def __init__(self, teacher_model, student_model, temperature=1.0):
        super(FeatureBasedDistillation, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.adaptation_layers = nn.ModuleList()

    def forward(self, x):
        teacher_features = self.extract_features(self.teacher_model, x)
        student_features = self.extract_features(self.student_model, x)

        total_loss = 0

        for i, (teacher_feat, student_feat) in enumerate(zip(teacher_features, student_features)):
            # Get the spatial and channel dimensions of the teacher feature map
            t_c, t_h, t_w = teacher_feat.shape[1], teacher_feat.shape[2], teacher_feat.shape[3]
            s_c = student_feat.shape[1]

            # Create adaptation layer if it doesn't exist
            if len(self.adaptation_layers) <= i:
                adaptation_layer = nn.Conv2d(s_c, t_c, kernel_size=1, bias=False)
                self.adaptation_layers.append(adaptation_layer.to(student_feat.device))

            # Adapt the student feature map
            adapted_student_feat = self.adaptation_layers[i](student_feat)
            
            # Interpolate adapted student feature map to match teacher's spatial dimensions
            s_feat_interpolated = F.interpolate(adapted_student_feat, size=(t_h, t_w), mode='bilinear', align_corners=False)
            
            # Calculate MSE loss
            loss = F.mse_loss(teacher_feat, s_feat_interpolated)
            total_loss += loss

        return total_loss

    def extract_features(self, model, x):
        features = []
        y = []
        for m in model.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in model.save else None)  # save output
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.SiLU, Conv)):
                features.append(x)
        return features

    def normalize(self, x):
        return F.normalize(x.view(x.size(0), -1), dim=1)

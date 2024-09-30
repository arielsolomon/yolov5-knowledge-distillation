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


    def forward(self, x):
        teacher_features = self.extract_features(self.teacher_model, x)
        student_features = self.extract_features(self.student_model, x)

        total_loss = 0  # Initialize total_loss

        for idx, (teacher_feat, student_feat) in enumerate(zip(teacher_features[:len(student_features)], student_features)):

            s_feat1 = F.interpolate(student_feat, size=(teacher_feat[-1], teacher_feat[-2]), mode='bilinear', align_corners=False)
            # Calculate MSE loss
            mse_loss = nn.MSELoss()
            loss = mse_loss(teacher_features_projected, student_feat)
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

    def feature_distillation_loss(self, teacher_features, student_features):
        loss = 0
        for t_feat, s_feat in zip(teacher_features, student_features):
            if t_feat.size() != s_feat.size():
                s_feat = F.interpolate(s_feat, size=t_feat.size()[2:], mode='bilinear', align_corners=False)
            loss += F.mse_loss(
                self.normalize(s_feat),
                self.normalize(t_feat.detach())
            )
        return loss

    def normalize(self, x):
        return F.normalize(x.view(x.size(0), -1), dim=1)

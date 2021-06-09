from dino_frame.DINO.DINO_augmentation import RandCrop_tu,Gaussiannoise_tu,Mirror_tu,Spatial_tansform_tu
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from dino_frame.network_architecture.neural_network import ClassficationNetwork
import dino_frame.DINO.vision_transformer as vits
from dino_frame.DINO.vision_transformer import DINOHead
import numpy as np
import random
from dino_frame.DINO.utils import MultiCropWrapper
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)
class global_trans1(nn.Module):
    def __init__(self, image_size):
        super(global_trans1, self).__init__()


        self.global_trans1 = nn.Sequential(
            RandCrop_tu(image_size=image_size, crop_size=(32, 128, 128)),
            RandomApply(
                Gaussiannoise_tu(image_size=(32, 128, 128), SNR=20),
                p=0.8
            ),
            RandomApply(
                Mirror_tu(image_size=(32, 128, 128)),
                p=0.8
            ),
            RandomApply(
                Spatial_tansform_tu(image_size=(32, 128, 128)),
                p=0.8
            ),
        )
    def forward(self,x):
        return self.global_trans1(x)

class global_trans2(nn.Module):
    def __init__(self, image_size):
        super(global_trans2, self).__init__()

        self.global_trans2 = nn.Sequential(
            RandCrop_tu(image_size=image_size, crop_size=(32, 128, 128)),
            RandomApply(
                Gaussiannoise_tu(image_size=(32, 128, 128), SNR=20),
                p=0.8
            ),
            RandomApply(
                Mirror_tu(image_size=(32, 128, 128)),
                p=0.8
            ),
            RandomApply(
                Spatial_tansform_tu(image_size=(32, 128, 128)),
                p=0.8
            ),
        )

    def forward(self, x):
        return self.global_trans2(x)

class local_trans(nn.Module):
    def __init__(self, image_size):
        super(local_trans, self).__init__()

        self.local_trans = nn.Sequential(
            RandCrop_tu(image_size=image_size, crop_size=(16, 64, 64)),
            RandomApply(
                Gaussiannoise_tu(image_size=(16, 64, 64), SNR=20),
                p=0.8
            ),
            RandomApply(
                Mirror_tu(image_size=(16, 64, 64)),
                p=0.8
            ),
            RandomApply(
                Spatial_tansform_tu(image_size=(16, 64, 64)),
                p=0.8
            ),
        )

    def forward(self, x):
        return self.local_trans(x)



class dino_teacher(ClassficationNetwork):
    def __init__(self, num_classes, deep_supervision, image_size, local_crop_number=8):
        super(dino_teacher, self).__init__()
        self.do_ds = False
        norm_cfg = 'BN'
        activation_cfg = 'ReLU'
        self.default_arch = 'vit_small'
        self.conv_op = nn.Conv3d
        self.norm_op = nn.BatchNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = (8, 16, 16)
        self.out_dim = 65536
        self.use_bn_in_head = False
        self.norm_last_layer = True
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.local_crop_number = local_crop_number
        self.teacher = vits.__dict__[self.default_arch](
            patch_size=self.patch_size,
        )
        self.embed_dim = self.teacher.embed_dim
        self.teacher = MultiCropWrapper(
            self.teacher,
            DINOHead(self.embed_dim, self.out_dim, self.use_bn_in_head),
        )
    def forward(self, x):
        print("teacher:",len(x))
        teacher_output = self.teacher(x)
        return teacher_output
class dino_student(ClassficationNetwork):
    def __init__(self, num_classes, deep_supervision, image_size, local_crop_number=8):
        super(dino_student, self).__init__()
        self.do_ds = False
        norm_cfg = 'BN'
        activation_cfg = 'ReLU'
        self.default_arch = 'vit_small'
        self.conv_op = nn.Conv3d
        self.norm_op = nn.BatchNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = (8, 16, 16)
        self.out_dim = 65536
        self.use_bn_in_head = False
        self.norm_last_layer = True
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.local_crop_number = local_crop_number
        self.student = vits.__dict__[self.default_arch](
            patch_size=self.patch_size, drop_path_rate=0.1,
        )

        self.embed_dim = self.student.embed_dim
        self.student = MultiCropWrapper(self.student, DINOHead(
            self.embed_dim,
            self.out_dim,
            use_bn=self.use_bn_in_head,
            norm_last_layer=self.norm_last_layer,
        ))

    def forward(self, x):
        student_output = self.student(x)
        return student_output

class DINO(ClassficationNetwork):
    def __init__(self, num_classes,deep_supervision,image_size,local_crop_number=8):
        super(DINO, self).__init__()
        self.do_ds = False
        norm_cfg = 'BN'
        activation_cfg = 'ReLU'
        self.default_arch = 'vit_small'
        self.conv_op = nn.Conv3d
        self.norm_op = nn.BatchNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = (8,16,16)
        self.out_dim = 65536
        self.use_bn_in_head = False
        self.norm_last_layer = True

        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.local_crop_number = local_crop_number
        self.global_trans1 = nn.Sequential(
            RandCrop_tu(image_size=image_size,crop_size=(32,128,128)),
            RandomApply(
            Gaussiannoise_tu(image_size=(32,128,128),SNR=20),
            p=0.8
            ),
            RandomApply(
                Mirror_tu(image_size=(32,128,128)),
                p=0.8
            ),
            RandomApply(
                Spatial_tansform_tu(image_size=(32,128,128)),
                p=0.8
            ),
        )
        self.global_trans2 = nn.Sequential(
            RandCrop_tu(image_size=image_size, crop_size=(32, 128, 128)),
            RandomApply(
                Gaussiannoise_tu(image_size=(32, 128, 128), SNR=20),
                p=0.8
            ),
            RandomApply(
                Mirror_tu(image_size=(32, 128, 128)),
                p=0.8
            ),
            RandomApply(
                Spatial_tansform_tu(image_size=(32, 128, 128)),
                p=0.8
            ),
        )
        self.local_trans = nn.Sequential(
            RandCrop_tu(image_size=image_size, crop_size=(16,64, 64)),
            RandomApply(
                Gaussiannoise_tu(image_size=(16,64,64), SNR=20),
                p=0.8
            ),
            RandomApply(
                Mirror_tu(image_size=(16,64, 64)),
                p=0.8
            ),
            RandomApply(
                Spatial_tansform_tu(image_size=(16,64,64)),
                p=0.8
            ),
        )
        self.student = vits.__dict__[self.default_arch](
            patch_size = self.patch_size,drop_path_rate = 0.1,
        )
        self.teacher =  vits.__dict__[self.default_arch](
            patch_size = self.patch_size,
        )
        self.embed_dim = self.student.embed_dim
        self.student = MultiCropWrapper(self.student, DINOHead(
            self.embed_dim,
            self.out_dim,
            use_bn=self.use_bn_in_head,
            norm_last_layer=self.norm_last_layer,
        ))
        self.teacher = MultiCropWrapper(
            self.teacher,
            DINOHead(self.embed_dim, self.out_dim, self.use_bn_in_head),
        )

    def forward(self, x):
        crops = []
        crops.append(self.global_trans1(x))
        crops.append(self.global_trans2(x))
        for _ in range(self.local_crop_number):
            crops.append(self.local_trans(x))
        teacher_output = self.teacher(crops[:2])
        student_output = self.student(crops)
        return teacher_output,student_output




class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        #dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) )

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

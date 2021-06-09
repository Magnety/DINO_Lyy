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
from dino_frame.DINO.utils import MultiCropWrapper,myMultiCropWrapper
def dino_teacher(num_classes, deep_supervision, image_size):
    default_arch = 'vit_small'
    patch_size = (8, 16, 16)
    out_dim = 65536
    use_bn_in_head = False
    norm_last_layer = True
    teacher = vits.__dict__[default_arch](
        patch_size=patch_size,
    )
    embed_dim =teacher.embed_dim
    teacher = myMultiCropWrapper(num_classes, deep_supervision, image_size,
        teacher,
        DINOHead(embed_dim, out_dim, use_bn_in_head),
    )
    return teacher
def dino_student(num_classes, deep_supervision, image_size):
    default_arch = 'vit_small'
    patch_size = (8, 16, 16)
    out_dim = 65536
    use_bn_in_head = False
    norm_last_layer = True
    student = vits.__dict__[default_arch](
        patch_size=patch_size,
        drop_path_rate=0.1,
    )
    embed_dim =student.embed_dim
    student = myMultiCropWrapper(num_classes, deep_supervision, image_size,
        student,
        DINOHead(embed_dim, out_dim, use_bn_in_head,norm_last_layer=norm_last_layer,),
    )
    return student
class teacher(ClassficationNetwork):
    def __init__(self, num_classes, deep_supervision, image_size, local_crop_number=8):
        super(teacher, self).__init__()
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

class student(ClassficationNetwork):
    def __init__(self, num_classes, deep_supervision, image_size, local_crop_number=8):
        super(student, self).__init__()
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

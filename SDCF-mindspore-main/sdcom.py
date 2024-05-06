import mindspore
import mindspore.nn as nn
from mindspore import Tensor, ops

from posixpath import pardir
import os, sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,pardir)

from resnet import resnet50
from operators import DyChLinear, apply_differentiable, gumbel_softmax
from utils import set_exist_attr, load_org_weights
from mnv2 import MobileNetV2


class PolicyNet(nn.Cell):
    def __init__(self, num_ch, num_classes, num_segment, temperature=5., dropout=0.8, pre_trained=''):
        super(PolicyNet, self).__init__()
        self.temperature = temperature
        self.feat_net = MobileNetV2()
        self.feat_dim = self.feat_net.last_channel

        self.feat_net.classifier = nn.SequentialCell(
            nn.Dropout(p=dropout),
            nn.Dense(self.feat_dim, num_classes)
        )
        self.gate = Gate(self.feat_dim, num_ch, num_segment=num_segment)

    def construct(self, x):
        feat_map, feat = self.feat_net.get_featmap(x)
        policy_logit = self.feat_net.classifier(feat)

        g_logits = self.gate(feat_map)
        y_soft, ret, index = gumbel_softmax(g_logits, tau=self.temperature)

        return feat, policy_logit, y_soft, ret, index
    
    def set_temperature(self, tau):
        self.temperature = tau
    
    def decay_temperature(self, decay_ratio=None):
        if decay_ratio is not None:
            self.temperature *= decay_ratio


class Gate(nn.Cell):
    def __init__(self, in_planes, num_ch, num_segment, hidden_dim=1024, spatial_size=49):
        super(Gate, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_segment = num_segment

        self.encoder = nn.SequentialCell(
            nn.Conv2d(in_planes, 64, kernel_size=1, stride=1, padding=0, has_bias=False),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Flatten(),
            nn.Dense(spatial_size*64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh()
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Dense(hidden_dim, num_ch)
    
    def construct(self, x):
        mid_feat = self.encoder(x)
        mid_feat = mid_feat.view(-1, self.num_segment, self.hidden_dim)
        _b, _t, _ = mid_feat.shape
        hx = ops.zeros((self.gru.num_layers, _b, self.hidden_dim), mindspore.float32) #.cuda()
        # self.gru.flatten_parameters()
        out, _ = self.gru(mid_feat, hx)

        out = self.fc(out.reshape(_b*_t, -1))
        return out


class SDCOM(nn.Cell):
    def __init__(self, num_classes=200, 
        num_segment=16, dropout=0.2,
        channel_list=[0, 0.25, 0.5, 0.75, 1.0]):
        super(SDCOM, self).__init__()
    # def __init__(self, cfg, random_init=True):
    #     super(SDCOM, self).__init__()
    #     net = cfg.MODEL.FEAT_NET
    #     dropout = cfg.TRAIN.DROPOUT
    #     pre_trained = cfg.MODEL.PRETRAINED
    #     feat_pretrained = cfg.MODEL.FEAT_PRETRAINED
    #     policy_pretrained = cfg.MODEL.POLICY_PRETRAINED
    #     num_classes = cfg.MODEL.NUM_CLASSES
    #     num_segment = cfg.DATASET.N_SEGMENT
    #     channel_list = list(cfg.MODEL.CHANNEL_LIST)

    #     self.cfg = cfg
        self.num_segment = num_segment  # number of frames
        self.channel_ratio = -1
        self.channel_choice = -1
        self.channel_list = channel_list
        self.ch_score = None
        self.num_classes = num_classes

        self.backbone = resnet50()
        self.last_dim = 2048

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.policy_net = PolicyNet(len(self.channel_list), num_classes, num_segment=self.num_segment, dropout=dropout)
        self.classifier = DyChLinear(self.last_dim, num_classes)
        self.cat_dim = self.last_dim+self.policy_net.feat_dim
        self.final_classifier = PoolingClassifier(
            input_dim=self.cat_dim,
            num_segments=self.num_segment,
            num_classes=num_classes,
            dropout=dropout
        )

        # /********** inference/training mode ******************/
        # 'supernet': supernet training
        # 'policy': policy network training
        # 'inference': dynamic inference
        self.set_stage('supernet')
        # self.set_stage(self.cfg.TRAIN.STAGE)
        self.set_module_channel_list()
            
        # if self.cfg.TRAIN.STAGE=='supernet':
        #     self.freeze_policynet()
        # else:
        #     self.freeze_backbone()

    def construct(self, x):
        # b, t, c, h, w = x.shape
        # x = x.reshape(-1, c, h, w)
        b, c, t, h, w = x.shape
        x = x.transpose(0,2,1,3,4).reshape(-1, c, h, w)
        identity = x

        if self.stage=='policy':
            policy_feat, policy_logit, g_soft, g_hard, g_idx = self.policy_net(x)
            self.set_module_channel_choice(g_hard)
            x = self.backbone.feature_forward(x)
            x = apply_differentiable(x, g_soft,self.channel_list, self.last_dim)
            x = self.avgpool(x)
            slim_feat = x.view(x.size(0), -1)
            slim_logit = self.classifier(slim_feat)
            # slim_logit = apply_differentiable(slim_logit, g_soft,self.channel_list, self.num_classes, logit=True)

            cat_feat = ops.cat([policy_feat, slim_feat], dim=-1)
            cat_feat_v = cat_feat.view(-1, self.num_segment, self.cat_dim)
            cat_logit, cat_pred = self.final_classifier(cat_feat_v)
            return cat_logit, cat_pred, policy_logit, slim_logit, g_hard, cat_feat
        elif self.stage=='supernet':
            assert self.channel_ratio >= 0, 'Please set valid channel ratio first.'
            x = self.backbone.feature_forward(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            out_logit = self.classifier(x)
            out_logit = out_logit.view(b,self.num_segment,-1).mean(dim=1)
            return out_logit
        elif self.stage=='inference':
            policy_feat, policy_logit, g_soft, g_hard, g_idx = self.policy_net(x)
            feat_list = []
            for idx, ch_idx in enumerate(list(g_idx)):
                r_ch = self.channel_list[ch_idx]
                img = identity[idx].unsqueeze(0)
                self.set_channel_ratio(r_ch)
                x = self.backbone.feature_forward(img)
                x = self.avgpool(x)
                slim_feat = x.view(x.shape[0], -1)
                if self.last_dim==slim_feat.shape[-1]:
                    pad_feat = slim_feat
                else:
                    pad_feat = ops.cat([slim_feat, ops.zeros((1,self.last_dim-slim_feat.shape[-1]), mindspore.float32)], axis=-1)
                feat_list.append(pad_feat)
            img_feat = ops.cat(feat_list, axis=0)
            cat_feat = ops.cat([policy_feat, img_feat], axis=-1)
            cat_feat = cat_feat.view(-1, self.num_segment, self.cat_dim)
            cat_logit, cat_pred = self.final_classifier(cat_feat)
            return cat_logit, cat_pred, g_hard, g_idx, policy_feat, img_feat
        elif self.stage=='static_inference':
            policy_feat, policy_logit, g_soft, g_hard, g_idx = self.policy_net(x)
            self.set_channel_ratio(0.25)
            x = self.backbone.feature_forward(x)
            x = self.avgpool(x)
            slim_feat = x.view(x.size(0), -1)
            pad_feat = ops.cat([slim_feat, ops.zeros(b*t,self.last_dim-slim_feat.size(-1)).cuda()], dim=-1)
            # feat_list.append(pad_feat)
            cat_feat = ops.cat([policy_feat, pad_feat], dim=-1)
            cat_feat = cat_feat.view(-1, self.num_segment, self.cat_dim)
            cat_logit, cat_pred = self.final_classifier(cat_feat)
            return cat_logit, cat_pred, g_hard, g_idx
        else:
            raise(KeyError, 'Not supported stage %s.' % self.stage)

    def set_stage(self, stage):
        self.stage=stage
        for m in self.cells():
            set_exist_attr(m, 'stage', stage)
    
    def set_module_channel_list(self):
        for n, m in self.cells_and_names():
            set_exist_attr(m, 'channel_list', self.channel_list)
    
    def set_module_channel_choice(self, channel_choice):
        self.channel_choice = channel_choice
        for n, m in self.cells_and_names():
            set_exist_attr(m, 'channel_choice', channel_choice)

    def set_channel_ratio(self, channel_ratio):
        # set channel ratio manually
        self.channel_ratio = channel_ratio
        for n, m in self.cells_and_names():
            set_exist_attr(m, 'channel_ratio', channel_ratio)

    def freeze_backbone(self):
        def fix_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        for name, param in self.backbone.cells_and_names():
            param.requires_grad = False
        self.backbone.apply(fix_bn)
        for name, param in self.classifier.cells_and_names():
            param.requires_grad = False

    def freeze_policynet(self):
        for name, param in self.policy_net.cells_and_names():
            param.requires_grad = False
        for name, param in self.final_classifier.cells_and_names():
            param.requires_grad = False

    def set_ch_score(self, score):
        setattr(self, 'ch_score', score)
    
    def get_optim_policies(self):
        return [{'params': self.policy_net.gate.parameters(), 'initial_lr': self.cfg.TRAIN.LR, 'lr_mult': self.cfg.TRAIN.GATE_LR, 'decay_mult': 1,
                 'name': "policy_gate"}] \
               + [{'params': self.policy_net.feat_net.parameters(), 'initial_lr': self.cfg.TRAIN.LR, 'lr_mult': self.cfg.TRAIN.POLICY_LR, 'decay_mult': 1,
                   'name': "policy_cnn"}] \
               + [{'params': self.backbone.parameters(), 'initial_lr': self.cfg.TRAIN.LR, 'lr_mult': 0.1, 'decay_mult': 1, 'name': "backbone_layers"}] \
               + [{'params': self.classifier.parameters(), 'initial_lr': self.cfg.TRAIN.LR, 'lr_mult': 1, 'decay_mult': 1, 'name': "backbone_fc"}] \
               + [{'params': self.final_classifier.parameters(), 'initial_lr': self.cfg.TRAIN.LR, 'lr_mult': self.cfg.TRAIN.CLS_FC_LR, 'decay_mult': 1,
                   'name': "pooling_classifier"}]


class MaxPooling(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x, y):
        x = ops.cat((x.unsqueeze(dim=1), y.unsqueeze(dim=1)), axis=1)
        return x.max(axis=1)


class MultiLayerPerceptron(nn.Cell):
    def __init__(self, input_dim, num_neurons=4096):
        super().__init__()
        self.input_dim = input_dim
        self.num_neurons = [num_neurons]
        layers = []
        dim_input = input_dim
        for dim_output in self.num_neurons:
            layers.append(nn.Dense(dim_input, dim_output))
            layers.append(nn.BatchNorm1d(dim_output))
            layers.append(nn.ReLU())
            dim_input = dim_output
        self.layers = nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.layers(x)
        return x


class PoolingClassifier(nn.Cell):
    def __init__(self, input_dim, num_segments, num_classes, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.max_pooling = MaxPooling()
        self.mlp = MultiLayerPerceptron(input_dim)
        self.num_segments = num_segments
        self.classifiers = nn.CellList()
        for m in range(self.num_segments):
            self.classifiers.append(nn.SequentialCell(
                nn.Dropout(p=dropout),
                nn.Dense(4096, self.num_classes)
            ))

    def construct(self, x):
        _b = x.shape[0]
        x = x.view(-1, self.input_dim)
        z = self.mlp(x).view(_b, self.num_segments, -1)
        logits = ops.zeros((_b, self.num_segments, self.num_classes), mindspore.float32) #.cuda()
        cur_z = z[:, 0]
        for frame_idx in range(0, self.num_segments):
            if frame_idx > 0:
                cur_z = self.max_pooling(z[:, frame_idx], cur_z)
            logits[:, frame_idx] = self.classifiers[frame_idx](cur_z)
        last_out = logits[:, -1, :].reshape(_b, -1)
        logits = logits.view(_b * self.num_segments, -1)
        return logits, last_out


if __name__ =='__main__':
    from mindspore import context
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    model = SDCOM(num_segment=16)
    # m.set_channel_ratio(1.0)
    model.set_stage('inference')
    d_in = ops.zeros((1,16,3,224,224), mindspore.float32)
    out = model(d_in)
    with open('cells.txt', 'w') as f:
        for name, cell in model.cells_and_names():
            f.write(name+'\n')
    # model_profiling(m, 224,224,16,3,use_cuda=False,verbose=True)

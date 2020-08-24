import glob
import math
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.optim import lr_scheduler
from criterion import RegnetLoss
from config import _C as config

def to_gpu(x, device):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(device, non_blocking=True)
    return torch.autograd.Variable(x)

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal
        
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(config.n_mel_channels, config.postnet_embedding_dim,
                         kernel_size=config.postnet_kernel_size, stride=1,
                         padding=int((config.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(config.postnet_embedding_dim))
        )

        for i in range(1, config.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(config.postnet_embedding_dim,
                             config.postnet_embedding_dim,
                             kernel_size=config.postnet_kernel_size, stride=1,
                             padding=int((config.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(config.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(config.postnet_embedding_dim, config.n_mel_channels,
                         kernel_size=config.postnet_kernel_size, stride=1,
                         padding=int((config.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(config.n_mel_channels))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))
        x = self.convolutions[-1](x)

        return x


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.random_z_dim = config.random_z_dim
        self.encoder_dim_with_z = config.visual_dim + self.random_z_dim

        convolutions = []
        for i in range(config.encoder_n_convolutions):
            conv_input_dim = self.encoder_dim_with_z if i==0 else config.encoder_embedding_dim
            conv_layer = nn.Sequential(
                ConvNorm(conv_input_dim,
                         config.encoder_embedding_dim,
                         kernel_size=config.encoder_kernel_size, stride=1,
                         padding=int((config.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(config.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.BiLSTM = nn.LSTM(config.encoder_embedding_dim,
                           int(config.encoder_embedding_dim / 4), config.encoder_n_lstm,
                           batch_first=True, bidirectional=True)
        self.BiLSTM_proj = nn.Linear(int(config.encoder_embedding_dim/2), int(config.encoder_embedding_dim/2))

    def forward(self, x):
        x = x.transpose(1, 2)
        z = torch.randn(x.shape[0], self.random_z_dim).to('cuda:0')
        z = z.view(z.size(0), z.size(1), 1).expand(z.size(0), z.size(1), x.size(2))
        x = torch.cat([x, z], 1)
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)
        x, _ = self.BiLSTM(x)
        x = self.BiLSTM_proj(x)
        return x


class Auxiliary_lstm_last(nn.Module):

    def __init__(self):
        super(Auxiliary_lstm_last, self).__init__()
        self.BiLSTM = nn.LSTM(config.n_mel_channels, int(config.auxiliary_dim), 2,
                           batch_first=True, bidirectional=True)
        self.BiLSTM_proj = nn.Linear(config.auxiliary_dim, config.auxiliary_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x, (h, c) = self.BiLSTM(x)
        x = self.BiLSTM_proj(h[-1])
        bs, c = x.shape
        x = x.unsqueeze(1).expand(bs, 215, c)
        return x


class Auxiliary_lstm_sample(nn.Module):

    def __init__(self):
        super(Auxiliary_lstm_sample, self).__init__()
        self.BiLSTM = nn.LSTM(config.n_mel_channels, int(config.auxiliary_dim/2), 2,
                           batch_first=True, bidirectional=True)
        self.auxiliary_sample_rate = config.auxiliary_sample_rate

    def forward(self, x):
        x = x.transpose(1, 2)
        x, (h, c) = self.BiLSTM(x)
        bs, T, C = x.shape
        forword = x[:, :, :int(C/2)]
        backword = x[:, :, int(C/2):]

        forword_sampled = forword[:, torch.arange(0, T, self.auxiliary_sample_rate).long(), :]
        backword_sampled = backword[:, torch.arange(0, T, self.auxiliary_sample_rate).long()+1, :]
        sampled = torch.cat([forword_sampled, backword_sampled], dim=-1)
        sampled_repeat = sampled.unsqueeze(1).repeat(1, int(self.auxiliary_sample_rate/4), 1, 1).view(bs, -1, C)
        assert sampled_repeat.shape[1] == math.ceil(860/self.auxiliary_sample_rate) * int(self.auxiliary_sample_rate/4)
        sampled_repeat = sampled_repeat[:, :215, :]
        return sampled_repeat


class Auxiliary_conv(nn.Module):

    def __init__(self):
        super(Auxiliary_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(config.n_mel_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.Conv1d(32, config.auxiliary_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(config.auxiliary_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.n_mel_channels = config.n_mel_channels
        model = []
        model += [nn.ConvTranspose1d(in_channels=config.decoder_conv_dim + config.auxiliary_dim, out_channels=int(config.decoder_conv_dim / 2),
                               kernel_size=4, stride=2, padding=1)]
        model += [nn.BatchNorm1d(int(config.decoder_conv_dim / 2))]
        model += [nn.ReLU(True)]

        model += [nn.Conv1d(in_channels=int(config.decoder_conv_dim / 2), out_channels=int(config.decoder_conv_dim / 2),
                               kernel_size=5, stride=1, padding=2)]
        model += [nn.BatchNorm1d(int(config.decoder_conv_dim / 2))]
        model += [nn.ReLU(True)]

        model += [nn.ConvTranspose1d(in_channels=int(config.decoder_conv_dim / 2), out_channels=self.n_mel_channels,
                               kernel_size=4, stride=2, padding=1)]
        model += [nn.BatchNorm1d(self.n_mel_channels)]
        model += [nn.ReLU(True)]

        model += [nn.Conv1d(in_channels=int(self.n_mel_channels), out_channels=self.n_mel_channels,
                               kernel_size=5, stride=1, padding=2)]

        self.model = nn.Sequential(*model)

    def forward(self, decoder_inputs):
        x = decoder_inputs.transpose(1, 2)

        x = self.model(x)
        return x


class Regnet_G(nn.Module):
    def __init__(self):
        super(Regnet_G, self).__init__()
        auxiliary_class = None
        if config.auxiliary_type == "lstm_last":
            auxiliary_class = Auxiliary_lstm_last
        elif config.auxiliary_type == "lstm_sample":
            auxiliary_class = Auxiliary_lstm_sample
        elif config.auxiliary_type == "conv":
            auxiliary_class = Auxiliary_conv
        self.n_mel_channels = config.n_mel_channels
        self.encoder = Encoder()
        self.auxiliary = auxiliary_class()
        self.decoder = Decoder()
        self.postnet = Postnet()
        if config.mode_input == "":
            self.mode_input = "vis_spec" if self.training else "vis"
        else:
            self.mode_input = config.mode_input
        self.aux_zero = config.aux_zero

    def forward(self, inputs, real_B):
        if self.mode_input == "vis_spec":
            vis_thr, spec_thr = 1, 1
        elif self.mode_input == "vis":
            vis_thr, spec_thr = 1, 0
        elif self.mode_input == "spec":
            vis_thr, spec_thr = 0, 1
        else:
            print(self.mode_input)
        encoder_output = self.encoder(inputs * vis_thr)
        gt_auxilitary = self.auxiliary(real_B * spec_thr)
        if self.aux_zero:
            gt_auxilitary = gt_auxilitary * 0
        encoder_output = torch.cat([encoder_output, gt_auxilitary], dim=2)
        mel_output_decoder = self.decoder(encoder_output)
        mel_output_postnet = self.postnet(mel_output_decoder)
        mel_output = mel_output_decoder + mel_output_postnet
        self.gt_auxilitary = gt_auxilitary
        return mel_output, mel_output_decoder


class Regnet_D(nn.Module):
    def __init__(self):
        super(Regnet_D, self).__init__()

        self.feature_conv = nn.Sequential(
            nn.ConvTranspose1d(config.visual_dim, config.decoder_conv_dim,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(config.decoder_conv_dim),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(config.decoder_conv_dim, 64,
                               kernel_size=4, stride=2, padding=1),
        )

        self.mel_conv = nn.ConvTranspose1d(config.n_mel_channels, 64,
                               kernel_size=1, stride=1)

        sequence = [
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(512, 1024, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(1024, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.down_sampling = nn.Sequential(*sequence)  # receptive field = 34

    def forward(self, *inputs):
        feature, mel = inputs
        feature_conv = self.feature_conv(feature.transpose(1, 2))
        mel_conv = self.mel_conv(mel)
        input_cat = torch.cat((feature_conv, mel_conv), 1)
        out = self.down_sampling(input_cat)
        out = nn.Sigmoid()(out)
        return out


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def init_net(net, device, init_type='normal', init_gain=0.02):
    assert (torch.cuda.is_available())
    net.to(device)
    net = torch.nn.DataParallel(net, range(torch.cuda.device_count()))
    init_weights(net, init_type, gain=init_gain)
    return net


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class Regnet(nn.Module):
    def __init__(self):
        super(Regnet, self).__init__()
        self.config = config
        self.n_mel_channels = config.n_mel_channels
        self.model_names = ['G', 'D']
        self.device = torch.device('cuda:0')
        self.netG = init_net(Regnet_G(), self.device)
        self.netD = init_net(Regnet_D(), self.device)
        self.criterionGAN = GANLoss().to(self.device)
        self.criterionL1 = RegnetLoss(config.loss_type).to(self.device)

        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=config.lr, betas=(config.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=config.lr, betas=(config.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.D_interval = config.D_interval
        self.n_iter = -1
        self.wo_G_GAN = config.wo_G_GAN

    def parse_batch(self, batch):
        input, mel, video_name = batch
        self.real_A = input.to(self.device).float()
        self.real_B = mel.to(self.device).float()
        self.video_name = video_name

    def forward(self):
        self.fake_B, self.fake_B_postnet = self.netG(self.real_A, self.real_B)

    def get_scheduler(self, optimizer, config):
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch + 2 - config.niter) / float(config.epochs - config.niter + 1)
            lr_l = 1.0 - max(0, epoch + 2 + config.epoch_count - config.niter) / float(config.epochs - config.niter + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        return scheduler

    def setup(self):
        self.schedulers = [self.get_scheduler(optimizer, config) for optimizer in self.optimizers]

    def load_checkpoint(self, checkpoint_path):
        for name in self.model_names:
            filepath = "{}_net{}".format(checkpoint_path, name)
            print("Loading net{} from checkpoint '{}'".format(name, filepath))
            state_dict = torch.load(filepath, map_location='cpu')
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            checkpoint_state = state_dict["optimizer_net{}".format(name)]
            net.load_state_dict(checkpoint_state)
            self.iteration = state_dict["iteration"]

            learning_rate = state_dict["learning_rate"]
        for index in range(len(self.optimizers)):
            for param_group in self.optimizers[index].param_groups:
                param_group['lr'] = learning_rate

    def save_checkpoint(self, save_directory, iteration):
        lr = self.optimizers[0].param_groups[0]['lr']
        for name in self.model_names:
            filepath = os.path.join(save_directory, "checkpoint_{:0>6d}_net{}".format(iteration, name))
            print("Saving net{} and optimizer state at iteration {} to {}".format(
                name, iteration, filepath))
            net = getattr(self, 'net' + name)
            if torch.cuda.is_available():
                torch.save({"iteration": iteration,
                            "learning_rate": lr,
                            "optimizer_net{}".format(name): net.module.cpu().state_dict()}, filepath)
                net.to(self.device)
            else:
                torch.save({"iteration": iteration,
                            "learning_rate": lr,
                            "optimizer_net{}".format(name): net.cpu().state_dict()}, filepath)

            """delete old model"""
            model_list = glob.glob(os.path.join(save_directory, "checkpoint_*_*"))
            model_list.sort()
            for model_path in model_list[:-2]:
                cmd = "rm {}".format(model_path)
                print(cmd)
                os.system(cmd)
        return model_list[-1][:-5]


    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.real_A.detach(), self.fake_B.detach())
        self.pred_fake = pred_fake.data.cpu()
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        pred_real = self.netD(self.real_A, self.real_B)
        self.pred_real = pred_real.data.cpu()
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        if not self.wo_G_GAN:
            pred_fake = self.netD(self.real_A, self.fake_B)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        else:
            self.loss_G_GAN = 0

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1((self.fake_B, self.fake_B_postnet), self.real_B)

        # Third, silence loss
        self.loss_G_silence = self.criterionL1((self.fake_B, self.fake_B_postnet), torch.zeros_like(self.real_B))

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.config.lambda_Oriloss + self.loss_G_silence * self.config.lambda_Silenceloss

        self.loss_G.backward()

    def optimize_parameters(self):
        self.n_iter += 1
        self.forward()
        # update D
        if self.n_iter % self.D_interval == 0:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
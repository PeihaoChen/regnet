import random
from torch.utils.tensorboard import SummaryWriter

class RegnetLogger(SummaryWriter):
    def __init__(self, logdir):
        super(RegnetLogger, self).__init__(logdir)

    def log_training(self, model, reduced_loss, learning_rate, duration,
                     iteration):
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("training.loss_G", model.loss_G, iteration)
        self.add_scalar("training.loss_D", model.loss_D, iteration)
        self.add_scalar("training.loss_G_GAN", model.loss_G_GAN, iteration)
        self.add_scalar("training.loss_G_L1", model.loss_G_L1, iteration)
        self.add_scalar("training.loss_G_silence", model.loss_G_silence, iteration)
        self.add_scalar("training.loss_D_fake", model.loss_D_fake, iteration)
        self.add_scalar("training.loss_D_real", model.loss_D_real, iteration)
        self.add_scalar("training.score_D_r-f", (model.pred_real - model.pred_fake).mean(), iteration)
        self.add_scalar("duration", duration, iteration)

    def log_testing(self, reduced_loss, epoch):
        self.add_scalar("testing.loss", reduced_loss, epoch)

    def log_plot(self, model, iteration, split="train"):
        output = model.fake_B
        output_postnet = model.fake_B_postnet
        target = model.real_B
        video_name = model.video_name

        idx = random.randint(0, output.size(0) - 1)

        self.add_image(
            "mel_spectrogram_{}".format(mode),
            plot_spectrogram(target[idx].data.cpu().numpy(),
                             output[idx].data.cpu().numpy(),
                             output_postnet[idx].data.cpu().numpy(),
                             video_name[idx], mode),
            iteration)
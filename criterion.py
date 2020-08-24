from torch import nn


class RegnetLoss(nn.Module):
    def __init__(self, loss_type):
        super(RegnetLoss, self).__init__()
        self.loss_type = loss_type
        print("Loss type: {}".format(self.loss_type))

    def forward(self, model_output, targets):

        mel_target = targets
        mel_target.requires_grad = False
        mel_out, mel_out_postnet = model_output

        if self.loss_type == "MSE":
            loss_fn = nn.MSELoss()
        elif self.loss_type == "L1Loss":
            loss_fn = nn.L1Loss()
        else:
            print("ERROR LOSS TYPE!")

        mel_loss = loss_fn(mel_out, mel_target) + \
                   loss_fn(mel_out_postnet, mel_target)

        return mel_loss
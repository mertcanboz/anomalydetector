"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation ("Microsoft") grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

configs = [()]


def make_layers(Bn=True, input=256):
    global configs
    layers = []
    layer = nn.Conv2d(input, input, kernel_size=1, stride=1, padding=0)
    layers.append(layer)
    if Bn:
        layers.append(nn.BatchNorm2d(input))

    for k, s, c in configs:
        if c == -1:
            layer = nn.Conv2d(kernel_size=k, stride=s, padding=0)
        else:
            now = []
            now.append(nn.Conv1d(input, c, kernel_size=k, stride=s, padding=0))
            input = c
            if Bn:
                now.append(nn.BatchNorm2d(input))
            now.append(nn.Relu(inplace=True))
            layer = nn.Sequential(*now)
        layers.append(layer)
    return nn.Sequential(*layers), input


class trynet(nn.Module):
    def __init__(self):
        super(trynet, self).__init__()
        self.layer1 = nn.Conv1d(1, 128, kernel_size=128, stride=0, padding=0)
        self.layer2 = nn.BatchNorm1d(128)

        self.feature = make_layers()


class Anomaly(nn.Module):
    def __init__(self, window=1024):
        self.window = window
        super(Anomaly, self).__init__()
        self.layer1 = nn.Conv1d(window, window, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv1d(window, 2 * window, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(2 * window, 4 * window)
        self.fc2 = nn.Linear(4 * window, window)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), self.window, 1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


def save_model(model, model_path):
    try:
        torch.save(model.state_dict(), model_path)
    except:
        torch.save(model, model_path)


def load_model(model, path):
    print("loading %s" % path)
    with open(path, 'rb') as f:
        pretrained = torch.load(f, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in pretrained.items() if k in model_dict}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)
    return model
  
class AbsPool(nn.Module):
    def __init__(self, pooling_module=None, *args, **kwargs):
        super(AbsPool, self).__init__()
        self.pooling_layer = pooling_module(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_pool = self.pooling_layer(x)
        neg_pool = self.pooling_layer(-x)
        abs_pool = torch.where(pos_pool >= neg_pool, pos_pool, -neg_pool)
        return abs_pool


MaxAbsPool1d = partial(AbsPool, nn.MaxPool1d)
MaxAbsPool2d = partial(AbsPool, nn.MaxPool2d)
MaxAbsPool3d = partial(AbsPool, nn.MaxPool3d)

class SharpenedCosineSimilarity(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int=3,
        padding: int='same',
        stride: int=1,
        groups: int=1,
        shared_weights: bool = False,
        log_p_init: float=.7,
        log_q_init: float=1.,
        log_p_scale: float=5.,
        log_q_scale: float=.3,
        alpha: Optional[float]=10,
        alpha_autoinit: bool=False,
        eps: float=1e-6,
    ):
        assert groups == 1 or groups == in_channels, " ".join([
            "'groups' needs to be 1 or 'in_channels' ",
            f"({in_channels})."])
        assert out_channels % groups == 0, " ".join([
            "The number of",
            "output channels needs to be a multiple of the number",
            "of groups.\nHere there are",
            f"{out_channels} output channels and {groups}",
            "groups."])

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.shared_weights = shared_weights

        if self.groups == 1:
            self.shared_weights = False

        super(SharpenedCosineSimilarity, self).__init__(
            self.in_channels,
            self.out_channels,
            kernel_size,
            bias=False,
            padding=padding,
            stride=stride,
            groups=self.groups)

        # Overwrite self.kernel_size created in the 'super' above.
        # We want an int, assuming a square kernel, rather than a tuple.
        self.kernel_size = kernel_size

        # Scaling weights in this way generates kernels that have
        # an l2-norm of about 1. Since they get normalized to 1 during
        # the forward pass anyway, this prevents any numerical
        # or gradient weirdness that might result from large amounts of
        # rescaling.
        self.channels_per_kernel = self.in_channels // self.groups
        weights_per_kernel = self.channels_per_kernel * self.kernel_size ** 2
        if self.shared_weights:
            self.n_kernels = self.out_channels // self.groups
        else:
            self.n_kernels = self.out_channels
        initialization_scale = (3 / weights_per_kernel) ** .5
        scaled_weight = np.random.uniform(
            low=-initialization_scale,
            high=initialization_scale,
            size=(
                self.n_kernels,
                self.channels_per_kernel,
                self.kernel_size)
        )
        self.weight = torch.nn.Parameter(torch.Tensor(scaled_weight))

        self.log_p_scale = log_p_scale
        self.log_q_scale = log_q_scale
        self.p = torch.nn.Parameter(torch.full(
            (1, self.n_kernels, 1),
            float(log_p_init * self.log_p_scale)))
        self.q = torch.nn.Parameter(torch.full(
            (1, 1, 1), float(log_q_init * self.log_q_scale)))
        self.eps = eps

        if alpha is not None:
            self.alpha = torch.nn.Parameter(torch.full(
                (1, 1, 1), float(alpha)))
        else:
            self.alpha = None
        if alpha_autoinit and (alpha is not None):
            self.LSUV_like_init()

    def LSUV_like_init(self, batch_size=BATCH_SIZE):
        BS, CH = batch_size, int(self.weight.shape[1]*self.groups)
        L = self.weight.shape[2]
        device = self.weight.device
        inp = torch.rand(BS, CH, L, device=device)
        with torch.no_grad():
            out = self.forward(inp)
            coef = (out.std(dim=(0, 2)) + self.eps).mean()
            self.alpha.data *= 1.0 / coef.view_as(self.alpha)
        return

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # Scale and transform the p and q parameters
        # to ensure that their magnitudes are appropriate
        # and their gradients are smooth
        # so that they will be learned well.
        p = torch.exp(self.p / self.log_p_scale)
        q = torch.exp(-self.q / self.log_q_scale)

        # If necessary, expand out the weight and p parameters.
        if self.shared_weights:
            weight = torch.tile(self.weight, (self.groups, 1, 1))
            p = torch.tile(p, (1, self.groups, 1))
        else:
            weight = self.weight

        return self.scs(inp, weight, p, q)

    def scs(self, inp, weight, p, q):
        # Normalize the kernel weights.
        weight = weight / self.weight_norm(weight)

        # Normalize the inputs and
        # Calculate the dot product of the normalized kernels and the
        # normalized inputs.
        cos_sim = F.conv1d(
            inp,
            weight,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        ) / self.input_norm(inp, q)

        # Raise the result to the power p, keeping the sign of the original.
        out = cos_sim.sign() * (cos_sim.abs() + self.eps) ** p

        # Apply learned scale parameter
        if self.alpha is not None:
            out = self.alpha.view(1, -1, 1) * out
        return out

    def weight_norm(self, weight):
        # Find the l2-norm of the weights in each kernel.
        return weight.square().sum(dim=(1, 2), keepdim=True).sqrt()

    def input_norm(self, inp, q):
        # Find the l2-norm of the inputs at each position of the kernels.
        # Sum the squared inputs over each set of kernel positions
        # by convolving them with the mock all-ones kernel weights.
        xnorm = F.conv1d(
            inp.square(),
            torch.ones((
                self.groups,
                self.channels_per_kernel,
                self.kernel_size)),
            stride=self.stride,
            padding=self.padding,
            groups=self.groups)

        # Add in the q parameter. 
        xnorm = (xnorm + self.eps).sqrt() + q
        outputs_per_group = self.out_channels // self.groups
        return torch.repeat_interleave(xnorm, outputs_per_group, axis=1)
      
n_input_channels = 1
n_units = 64
kernel_size = 1

class Network(nn.Module):
    def __init__(self, window_size=1024):
        super().__init__()
        self.window_size = window_size
        self.scs1 = SharpenedCosineSimilarity(
            in_channels=self.window_size,
            out_channels=self.window_size,
            kernel_size=kernel_size,
            groups=n_input_channels)
        self.pool1 = MaxAbsPool1d(kernel_size=2, stride=2, ceil_mode=True)

        self.scs2_depth = SharpenedCosineSimilarity(
            in_channels=self.window_size,
            out_channels=self.window_size*2,
            kernel_size=kernel_size)
        self.scs2_point = SharpenedCosineSimilarity(
            in_channels=n_units,
            out_channels=n_units,
            kernel_size=1)
        self.pool2 = MaxAbsPool1d(kernel_size=2, stride=2, ceil_mode=True)
        
        self.fc1 = nn.Linear(in_features=self.window_size*2, out_features=self.window_size*4)
        self.fc2 = nn.Linear(in_features=self.window_size*4, out_features=self.window_size)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = x.view(x.size(0), self.window_size, 1)
        x = self.scs1(x)
        x = self.scs2_depth(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

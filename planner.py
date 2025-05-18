import torch
import torch.nn.functional as F


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class Planner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Create a direct CNN architecture instead of using ResNet50
        print("Creating custom CNN planner...")
        
        # Input shape: (B, 3, 96, 128)
        self.cnn = torch.nn.Sequential(
            # First block - reduce dimensions to half
            torch.nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            # 32x48x64
            
            # Second block - reduce dimensions to 1/4
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # 64x24x32
            
            # Third block
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # 128x24x32
            
            # Fourth block - reduce dimensions to 1/8
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # 256x12x16
            
            # Fifth block
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # 256x12x16
            
            # Final layers to produce heatmap
            torch.nn.Conv2d(256, 128, kernel_size=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(64, 1, kernel_size=1)
            # Output: 1x12x16 heatmap for spatial_argmax
        )
        
        print("Custom CNN planner created successfully")

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        # Pass input directly through CNN
        x = self.cnn(img)
        
        # Apply spatial argmax to get coordinates
        return spatial_argmax(x[:, 0])


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from controller import control
    from utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)

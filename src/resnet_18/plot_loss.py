import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plot_utils import plot_losses

if __name__ == "__main__":
    plot_losses("output/resnet_18")

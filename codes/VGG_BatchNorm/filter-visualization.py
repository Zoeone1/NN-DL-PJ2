import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from models.vgg import VGG_A, VGG_A_BatchNorm, VGG_A_Dropout, VGG_A_Light

plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of negative sign display


def visualize_filters(model, layer_name, num_filters=64, save_path=None):
    layer = dict(model.named_modules())[layer_name]
    filters = layer.weight.data.cpu().numpy()
    filters = (filters - filters.min()) / (filters.max() - filters.min())  # Normalize to [0, 1]

    n = int(np.ceil(np.sqrt(num_filters)))
    fig, axes = plt.subplots(n, n, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(min(num_filters, len(filters))):
        filter_img = filters[i]

        # Dynamically handle filters with different dimensions (3D or 4D)
        if filter_img.ndim == 3:
            # Shape: (out_channels, in_channels, kernel_size)
            # Transpose to (kernel_size, in_channels, out_channels) -> Adjust to (H, W, C) format
            filter_img = filter_img.transpose(1, 2, 0)
        elif filter_img.ndim == 4:
            # Shape: (out_channels, in_channels, kernel_h, kernel_w)
            # Transpose to (kernel_h, kernel_w, in_channels)
            filter_img = filter_img.transpose(2, 3, 1)
        else:
            print(f"Warning: Unsupported filter dimension {filter_img.ndim}, skipping visualization")
            continue

        # Handle multi-channel cases (number of input channels may be greater than 3)
        if filter_img.shape[-1] > 3:
            # Average over multiple channels to convert to a single-channel grayscale image
            filter_img = np.mean(filter_img, axis=-1)
            cmap = 'viridis'  # Use a clearer colormap
        else:
            cmap = 'gray' if filter_img.shape[-1] == 1 else None  # Use grayscale for single-channel

        # Ensure pixel values are within the valid range
        filter_img = np.clip(filter_img, 0, 1)

        axes[i].imshow(filter_img, cmap=cmap)
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i + 1}')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Filter visualization saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_all_conv_layers(model, save_dir='filter_visualizations'):
    os.makedirs(save_dir, exist_ok=True)
    conv_layers = [name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    print(f"Found {len(conv_layers)} convolutional layers: {conv_layers}")

    for i, layer_name in enumerate(conv_layers):
        print(f"Visualizing convolutional layer: {layer_name}")
        save_path = os.path.join(save_dir, f'filters_{layer_name.replace(".", "_")}.png')
        visualize_filters(model, layer_name, save_path=save_path)


def main():
    model = VGG_A_Dropout()
    try:
        model.load_state_dict(torch.load('VGG_A_Dropout_best_model-L2loss.pth'))
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error: {e}")
        return

    model.eval()
    visualize_all_conv_layers(model)


if __name__ == "__main__":
    main()
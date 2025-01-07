import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import sys
import os
import seaborn as sns
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import find_vertical_line_bounds

def create_scientific_pipeline_vis(img_path, output_path):
    """Create a scientific visualization of the preprocessing pipeline with arrows."""
    sns.set_theme('paper', 'white')

    # Create figure
    fig = plt.figure(figsize=(15, 6))  # Reduced to 3x2 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Create subplots
    axes = []
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)
    
    # Read image
    original = cv2.imread(img_path)
    if original is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    # Process and plot each step
    steps = []
    
    # 1. Original Image
    steps.append({
        'image': cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
        'title': '(a) Original Image',
        'cmap': None
    })
    
    # 2. Grayscale + Gaussian Blur
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    steps.append({
        'image': blurred,
        'title': '(b) Grayscale + Blur',
        'cmap': 'gray'
    })
    
    # 3. Adaptive Threshold
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=13, C=12
    )
    steps.append({
        'image': binary,
        'title': '(c) Adaptive Threshold',
        'cmap': 'gray'
    })
    
    # 4. Morphological Opening
    kernel = np.ones((3,3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    steps.append({
        'image': opened,
        'title': '(d) Morphological Opening',
        'cmap': 'gray'
    })
    
    # 5. ROI Detection
    lb, rb = find_vertical_line_bounds(opened)
    final = opened.copy()
    final[:, :lb] = 255
    final[:, rb:] = 255
    steps.append({
        'image': final,
        'title': '(e) Region of Interest',
        'cmap': 'gray'
    })
    
    # 6. Resized
    target_size = (original.shape[1] // 2, original.shape[0] // 2)
    resized = cv2.resize(final, target_size, interpolation=cv2.INTER_AREA)
    stacked = np.stack([resized, resized, resized], axis=-1)
    steps.append({
        'image': stacked,
        'title': '(f) Resized',
        'cmap': None
    })
    
    # Plot all steps
    for ax, step in zip(axes, steps):
        ax.imshow(step['image'], cmap=step['cmap'])
        ax.set_title(step['title'], pad=10, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(.5)
            spine.set_edgecolor('black')
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
    
    # Add arrows between horizontally adjacent subplots
    def add_arrow(fig, ax1, ax2):
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        arrow = FancyArrowPatch(
            (pos1.x1, pos1.y0 + pos1.height/2),
            (pos2.x0, pos2.y0 + pos2.height/2),
            transform=fig.transFigure,
            connectionstyle="arc3,rad=0",
            arrowstyle='->',
            color='black',
            linewidth=1,
            mutation_scale=15
        )
        fig.add_artist(arrow)
    
    # Add horizontal arrows
    for i in range(len(axes)-1):
        if i % 3 != 2:  # Skip arrows between rows
            add_arrow(fig, axes[i], axes[i+1])
    
    # Add main title
    fig.suptitle('Preprocessing Pipeline', y=0.95, fontsize=12, fontweight='bold')
    
    # Save
    plt.savefig(output_path, format='png', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()

# create_scientific_pipeline_vis('/home/rmw874/piratbog/data/raw/temp/8013620831-0030.jpg-t.jpg', 'preprocessing_pipeline_buggy_raw.pdf')

create_scientific_pipeline_vis('/home/rmw874/piratbog/data/processed/Mathiesen-single-pages/top/8013620831-0031.jpg-t.jpg', 'preprocessing_pipeline.png')

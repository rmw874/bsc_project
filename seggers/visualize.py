import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class ImageVisualizer:
    def __init__(self, folder1, folder2):
        self.images1 = sorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('png', 'jpg', 'jpeg'))])
        self.images2 = sorted([os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('png', 'jpg', 'jpeg'))])
        self.index = 0
        
        if len(self.images1) != len(self.images2):
            raise ValueError("The folders must have the same number of images!")
        
        self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update_display()

    def update_display(self):
        img1 = cv2.cvtColor(cv2.imread(self.images1[self.index]), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(self.images2[self.index]), cv2.COLOR_BGR2RGB)

        self.axs[0].clear()
        self.axs[0].imshow(img1)
        self.axs[0].set_title(f"Folder 1: {os.path.basename(self.images1[self.index])}")
        self.axs[0].axis('off')

        self.axs[1].clear()
        self.axs[1].imshow(img2)
        self.axs[1].set_title(f"Folder 2: {os.path.basename(self.images2[self.index])}")
        self.axs[1].axis('off')

        plt.draw()

    def on_key(self, event):
        if event.key == "right" and self.index < len(self.images1) - 1:
            self.index += 1
            self.update_display()
        elif event.key == "left" and self.index > 0:
            self.index -= 1
            self.update_display()

# Example usage
folder1 = "/Users/sofusbjorn/piratbog/data/Mathiesen-single-pages"  # Replace with the path to your first folder
folder2 = "/Users/sofusbjorn/piratbog/cropped_data"  # Replace with the path to your second folder

visualizer = ImageVisualizer(folder1, folder2)
plt.show()
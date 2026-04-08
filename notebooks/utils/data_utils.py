import numpy as np
import struct
from array import array
import matplotlib.pyplot as plt

class MnistDataHelper(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)
    
    def add_gaussian_noise(x, noise_factor=0.5):
        noise = np.random.normal(loc=0.0, scale=1.0, size=x.shape)
        x_noisy = x + noise_factor * noise
        return np.clip(x_noisy, 0., 1.)
    
    def plot_transformation(x_test, n_samples, networks):
        batch_inputs = x_test[np.random.randint(0, x_test.shape[0], size=n_samples)]
        
        for network in networks:
            outputs = network.forward(batch_inputs)
            outputs.insert(0, batch_inputs)
            
            n_stages = len(outputs)

            fig, axes = plt.subplots(n_samples, n_stages, figsize=(n_stages*2, n_samples*2), squeeze=False)

            for row in range(n_samples):
                for col, data in enumerate(outputs):
                    ax = axes[row, col] 

                    img = data[row] 

                    if img.size == 784:
                        img = img.reshape(28, 28)
                    else:
                        side = int(np.sqrt(img.size))
                        if side * side == img.size:
                            img = img.reshape(side, side)
                        else:
                            img = img.reshape(1, -1)

                    ax.imshow(img, cmap="gray")
                    ax.axis("off")

            plt.tight_layout()
            plt.show()
    
    def plot_net_comparison(x_test, n_samples, networks, names):
        batch_inputs = x_test[np.random.randint(0, x_test.shape[0], size=n_samples)]
        batch_outputs = [batch_inputs]
        names.insert(0, "Input")
        
        for network in networks:
            outputs = network.forward(batch_inputs)
            batch_outputs.append(outputs[-1])
        
        n_images = len(batch_outputs)
        
        fig, axes = plt.subplots(n_samples, n_images, figsize=(n_images*2, n_samples*2), squeeze=False)

        for row in range(n_samples):
            for col, output in enumerate(batch_outputs):        
                ax = axes[row, col] 
                img = output[row] 
                
                if row == 0:
                    ax.set_title(names[col], fontsize=14, fontweight='bold')

                if img.size == 784:
                    img = img.reshape(28, 28)
                else:
                    side = int(np.sqrt(img.size))
                    if side * side == img.size:
                        img = img.reshape(side, side)
                    else:
                        img = img.reshape(1, -1)

                ax.imshow(img, cmap="gray")
                ax.axis("off")

        plt.tight_layout()
        plt.show()
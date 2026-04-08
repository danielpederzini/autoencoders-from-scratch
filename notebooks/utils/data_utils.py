import numpy as np
import struct
from array import array
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sns


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
    
    def add_dead_pixels(x, noise_factor=0.5):
        noise = np.random.choice([0, 1], size=x.shape, p=[noise_factor, 1 - noise_factor])
        return x * noise
        
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

        plot_names = ["Input"]
        for name in names:
            plot_names.extend([f"{name}\nRecon", f"{name}\nError"])

        batch_outputs = [batch_inputs]
        for network in networks:
            outputs = network.forward(batch_inputs)[-1]
            batch_outputs.append(outputs)

        n_models = len(networks)
        n_images = 1 + 2 * n_models

        fig, axes = plt.subplots(
            n_samples,
            n_images,
            figsize=(n_images * 2.2, n_samples * 2.2),
            squeeze=False
        )

        for row in range(n_samples):
            input_img = batch_inputs[row]

            ax = axes[row, 0]
            img = input_img
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
            if row == 0:
                ax.set_title("Input", fontsize=14, fontweight="bold")

            for i, network_output in enumerate(batch_outputs[1:], start=0):
                recon = network_output[row]
                error = np.abs(input_img - recon)

                recon_col = 1 + 2 * i
                err_col = 2 + 2 * i

                ax = axes[row, recon_col]
                img = recon
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
                if row == 0:
                    ax.set_title(f"{names[i]}\nRecon", fontsize=14, fontweight="bold")

                ax = axes[row, err_col]
                img = error
                if img.size == 784:
                    img = img.reshape(28, 28)
                else:
                    side = int(np.sqrt(img.size))
                    if side * side == img.size:
                        img = img.reshape(side, side)
                    else:
                        img = img.reshape(1, -1)

                ax.imshow(img, cmap="hot")
                ax.axis("off")
                if row == 0:
                    ax.set_title(f"{names[i]}\nError", fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.show()
        
    def plot_anomaly_model_comparison(
        y_true,
        y_scores_list,
        y_preds_list,
        model_names,
        class_names=("Normal", "Anomaly")
    ):
        n_models = len(model_names)

        side_by_side = n_models >= 4

        fig = plt.figure(figsize=(5*n_models if not side_by_side else 18, 16))
        gs = fig.add_gridspec(3, n_models if not side_by_side else 2)

        for i in range(n_models):
            ax = fig.add_subplot(gs[0, i] if not side_by_side else fig.add_gridspec(3, n_models)[0, i])
            cm = confusion_matrix(y_true, y_preds_list[i])

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax
            )

            ax.set_title(f"{model_names[i]}\nConfusion Matrix", fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        if side_by_side:
            ax_roc = fig.add_subplot(gs[1, 0])
        else:
            ax_roc = fig.add_subplot(gs[1, :])

        for i in range(n_models):
            fpr, tpr, _ = roc_curve(y_true, y_scores_list[i])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f"{model_names[i]} (AUC={roc_auc:.3f})")

        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.set_title("ROC Curves", fontweight="bold")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()

        if side_by_side:
            ax_err = fig.add_subplot(gs[1, 1])
        else:
            ax_err = fig.add_subplot(gs[2, :])

        for i in range(n_models):
            normal_errors = y_scores_list[i][y_true == 0]
            anomaly_errors = y_scores_list[i][y_true == 1]

            sns.kdeplot(normal_errors, label=f"{model_names[i]} normal", ax=ax_err)
            sns.kdeplot(anomaly_errors, linestyle="--", label=f"{model_names[i]} anomaly", ax=ax_err)

        ax_err.set_title("Reconstruction Error Distributions", fontweight="bold")
        ax_err.set_xlabel("Reconstruction Error")
        ax_err.legend()
        
        plt.tight_layout()
        plt.show()
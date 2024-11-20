import os
import struct
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_mnist_images(file_path):
    """
    Load MNIST images from a binary file.
    """
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_mnist_labels(file_path):
    """
    Load MNIST labels from a binary file.
    """
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def generate_control(image):
    """
    Generate a control image set
    """
    image = Image.fromarray(image)
    return np.array(image)

# def generate_rts(image):
#     """
#     Generate a Rotated, Translated, and Scaled (RTS) images.
#     """
#     image = Image.fromarray(image)

#     # Random rotation between +45° and -45°
#     angle = np.random.uniform(-45, 45)
#     image = image.rotate(angle, expand=True, fillcolor=0)

#     # Random scaling between 0.7 and 1.2
#     scale = np.random.uniform(0.7, 1.2)
#     width, height = image.size
#     new_size = (int(width * scale), int(height * scale))
#     image = image.resize(new_size, Image.Resampling.LANCZOS)

#     # Ensure the scaled image fits within the canvas size
#     canvas_size = (42, 42)
#     if new_size[0] > canvas_size[0] or new_size[1] > canvas_size[1]:
#         # Resize the image to fit within the canvas
#         scale = min(canvas_size[0] / width, canvas_size[1] / height)
#         new_size = (int(width * scale), int(height * scale))
#         image = image.resize(new_size, Image.Resampling.LANCZOS)

#     # Place in a random position on the canvas
#     canvas = Image.new("L", canvas_size, (0,))
#     x_offset = np.random.randint(0, canvas_size[0] - new_size[0] + 1)
#     y_offset = np.random.randint(0, canvas_size[1] - new_size[1] + 1)
#     canvas.paste(image, (x_offset, y_offset))

#     return np.array(canvas)

def generate_rts(image):
    """
    Generate a Rotated, Translated, and Scaled (RTS) images.
    """
    image = Image.fromarray(image)

    # Random rotation between +45° and -45°
    angle = np.random.uniform(-45, 45)
    image = image.rotate(angle, expand=True, fillcolor=0)

    # Random scaling between 0.7 and 1.2
    scale = np.random.uniform(0.7, 1.2)
    width, height = image.size
    new_size = (int(width * scale), int(height * scale))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Ensure the scaled image fits within the canvas size
    canvas_size = (42, 42)
    if new_size[0] > canvas_size[0] or new_size[1] > canvas_size[1]:
        # Resize the image to fit within the canvas
        scale = min(canvas_size[0] / width, canvas_size[1] / height)
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Place in a random position on the canvas
    canvas = Image.new("L", canvas_size, (0,))
    x_offset = np.random.randint(0, canvas_size[0] - new_size[0] + 1)
    y_offset = np.random.randint(0, canvas_size[1] - new_size[1] + 1)
    canvas.paste(image, (x_offset, y_offset))

    return np.array(canvas)


def generate_p(image):
    """
    Generate a Projected (P) images with random corner distortions.
    """
    image = Image.fromarray(image)

    # Random scaling between 0.75 and 1.0
    scale = np.random.uniform(0.75, 1.0)
    width, height = image.size
    new_size = (int(width * scale), int(height * scale))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Stretch each corner by a random amount sampled from N(0, 5)
    width, height = new_size
    corners = [
        (0 + np.random.normal(0, 5), 0 + np.random.normal(0, 5)),  # Top-left
        (width + np.random.normal(0, 5), 0 + np.random.normal(0, 5)),  # Top-right
        (0 + np.random.normal(0, 5), height + np.random.normal(0, 5)),  # Bottom-left
        (width + np.random.normal(0, 5), height + np.random.normal(0, 5)),  # Bottom-right
    ]

    # Apply perspective transformation
    image = image.transform(
        (width, height),
        Image.QUAD,
        (corners[0][0], corners[0][1], corners[1][0], corners[1][1],
         corners[3][0], corners[3][1], corners[2][0], corners[2][1]),
        resample=Image.Resampling.BICUBIC,  # Updated to Resampling.BICUBIC
    )
    return np.array(image)


def visualize_mnist(images, labels, num_samples=10):
    """
    Visualize a specified number of MNIST images with their labels.
    """
    plt.figure(figsize=(10, 1))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_as_matrices(image, folder, file_name):
    """
    Save the image as a normalized matrix in the specified folder.
    """
    os.makedirs(folder, exist_ok=True)
    normalized_image = image / 255.0  # Normalize to the range [0, 1]
    np.save(os.path.join(folder, file_name), normalized_image)

# Define the paths to the files
data_folder = "MNIST_ORG"  # Assuming MNIST_ORG is the folder containing the files
train_images_path = os.path.join(data_folder, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(data_folder, 'train-labels.idx1-ubyte')

# Load the MNIST data
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)

# Create directories for RTS and P datasets as matrices
os.makedirs("control_matrices", exist_ok=True)
os.makedirs("RTS_matrices", exist_ok=True)
os.makedirs("P_alt_matrices", exist_ok=True)
os.makedirs("control_images", exist_ok=True)
os.makedirs("RTS_images", exist_ok=True)
os.makedirs("P_alt_images", exist_ok=True)

# Process and save RTS and P datasets
for i, image in enumerate(train_images):
    # Generate control, RTS, and P versions of the image
    control_image = generate_control(image)
    rts_image = generate_rts(image)
    p_image = generate_p(image)

    # Save images as matrices
    save_as_matrices(control_image, "control_matrices", f"{i}_label{train_labels[i]}.npy")
    save_as_matrices(rts_image, "RTS_matrices", f"{i}_label{train_labels[i]}.npy")
    save_as_matrices(p_image, "P_alt_matrices", f"{i}_label{train_labels[i]}.npy")

    # Save images as PNGs
    Image.fromarray(control_image).save(f"control_images/{i}_label{train_labels[i]}.png")
    Image.fromarray(rts_image).save(f"RTS_images/{i}_label{train_labels[i]}.png")
    Image.fromarray(p_image).save(f"P_alt_images/{i}_label{train_labels[i]}.png")

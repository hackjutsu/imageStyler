import tensorflow as tf
import numpy as np
import config
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("Is GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
print("Available GPU devices:", tf.config.list_physical_devices('GPU'))

# Function to load an image and limit its maximum dimension to 512 pixels
def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

# Function to display an image
def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

# Function to load VGG19 and retrieve the intermediate layers
def get_model():
  # Load our model. We load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False

  # Select the layers to use for style and content
  style_outputs = [vgg.get_layer(name).output for name in style_layers]
  content_outputs = [vgg.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs

  # Build the model
  return tf.keras.Model(vgg.input, model_outputs)

# Function to compute the content loss
def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))

# Function to compute the gram matrix for an input layer
def gram_matrix(input_tensor):
  channels = int(input_tensor.shape[-1])
  a = tf.reshape(input_tensor, [-1, channels])
  n = tf.shape(a)[0]
  gram = tf.matmul(a, a, transpose_a=True)
  return gram / tf.cast(n, tf.float32)

# Function to compute the style loss
def get_style_loss(base_style, gram_target):
  gram_style = gram_matrix(base_style)
  return tf.reduce_mean(tf.square(gram_style - gram_target))

# Function to get the style and content feature representations
def get_feature_representations(model, content_path, style_path):
  # Load our images in
  content_image = load_img(content_path)
  style_image = load_img(style_path)

  # batch compute content and style features
  style_outputs = model(style_image)
  content_outputs = model(content_image)

  # Get the style and content feature representations from our model
  style_features = [style_layer[0] for style_layer in style_outputs[:len(style_layers)]]
  content_features = [content_layer[0] for content_layer in content_outputs[len(style_layers):]]
  return style_features, content_features

# Convert tensor to image
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)

# For updating the image
@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    # Extract style and content features
    outputs = model(image)
    style_features = outputs[:num_style_layers]
    content_features = outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(style_gram_matrices, style_features):
      style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # Accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_features):
      content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)

    # Get total loss
    total_loss = style_score + content_score

  # Apply gradients to update the image
  grad = tape.gradient(total_loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

# Define the layers to use for style and content
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

content_layers = ['block5_conv2']

# Load content and style images
content_path = config.CONTENT_PATH
style_path = config.STYLE_PATH

# Load and preprocess the images
content_image = load_img(content_path)
style_image = load_img(style_path)

# Display the images
plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')

# Load the model
model = get_model()

# Get the style and content feature representations (from our specified intermediate layers)
style_features, content_features = get_feature_representations(model, content_path, style_path)
style_gram_matrices = [gram_matrix(style_feature) for style_feature in style_features]

# Set initial image
initial_image = load_img(content_path)
initial_image = tf.Variable(initial_image, dtype=tf.float32)

# Create our optimizer
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# For displaying intermediate images
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Now we run the style transfer
epochs = config.EPOCHS
steps_per_epoch = config.STEPS_PER_EPOCH

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(initial_image)
    print(".", end='')
  print("Train step: {}".format(step))

# Save the result
file_name = config.OUTPUT_PATH
tensor_to_image(initial_image).save(file_name)

# Display the stylized image
plt.figure(figsize=(14, 4))
plt.imshow(tensor_to_image(initial_image))
plt.title("Stylized Image")
plt.show()

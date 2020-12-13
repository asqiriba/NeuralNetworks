# -*- coding: utf-8 -*-
"""Final Pt2

# 2. (15 points) Use Keras to load one of the deep CNN models trained on the **ILSVRC-2012-CLS** image classification dataset (ImageNet). See below to find out which model to load:
> If your last name starts with A, B: InceptionResNetV2.
If your last name starts with C, D, E, F, G, H: ResNet50-V2.
If your last name starts with I, J, K, L, M: VGG16.
If your last name starts with N, O, P, Q: MobileNet-v2.
If your last name starts with R, S: InceptionV3.
If your last name starts with T, U, V, W, X, Y, Z: VGG19.

(Note: if you load the wrong network, you won’t get any points!)
"""

"""Last name: Asqiriba -> InceptionResNetV2

Keras class parameters:
  tf.keras.applications.InceptionResNetV2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs
  )

Atributes:
  IMG_SIZE   = 299x299
  Weights    = 215MB
  Top1A      = 0.804
  Top5A      = 0.953
  Parameters = 55,873,736
  Depth      = 572
  Accuracy   = 0.843
  F1 score   = 0.833

Keras implementation site:
  https://keras.io/api/applications/inceptionresnetv2/

InceptionResNetV2 site:
  https://arxiv.org/abs/1602.07261
"""
from keras.applications.inception_resnet_v2 import InceptionResNetV2, decode_predictions, preprocess_input

# Image dictionary stored in my Colab instance.
IMG_BANK = {
    'Lion' : '/content/download (1).jpeg',
    'Bike' : '/content/download (2).jpeg',
    'Coffee' : '/content/download (5).jpeg',
    'Kite' : '/content/download (6).jpeg',
    'Snail' : '/content/download (7).jpeg',
    'Bench' : '/content/download (8).jpeg',
    'Leopard' : '/content/download.jpeg',
    'Cat' : '/content/image.jpeg',
    'Bus' : '/content/images (1).jpeg',
    'Pig' : '/content/img.png'
}

# We store the features(mean) of the images for part C.
IMG_FEATURES = {
    'Lion' : 0,
    'Bike' : 0,
    'Coffee' : 0,
    'Kite' : 0,
    'Snail' : 0,
    'Bench' : 0,
    'Leopard' : 0,
    'Cat' : 0,
    'Bus' : 0,
    'Pig' : 0
}

# Init the model.
model = InceptionResNetV2(weights="imagenet")
print('Model: InceptionResNetV2')

"""## (a) Evaluate the performance of the selected model using the given test set available under Week 13 module (img.zip folder includes 10 images), and calculate the top1-accuracy and top5-accuracy for that."""

from keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Retrieve an image from the files.
IMG_PATH = IMG_BANK['Lion']

# Save the name of selected image for dict handling.
name_image = list(IMG_BANK.keys())[list(IMG_BANK.values()).index(IMG_PATH)]

# Load image, resize acording to the model's needs.
img = load_img(IMG_PATH, target_size=(299, 299))

# Convert img to arr(h, w, channel), then into batch(batch, h, w, ch).
processed_image = img_to_array(img)
batch_image = np.expand_dims(processed_image, axis=0)
digested_image = preprocess_input(batch_image)

# Digest image into the model and decode it; make it human-readable.
predictions = model.predict(digested_image)

# We take the TopK from the model, then print it.
top1_labels = decode_predictions(predictions, top=1)
top5_labels = decode_predictions(predictions)

# Brief explanation of what TopX accuracy means.
print('\nInceptionResNetV2 TopK scores:\n'
    + '\nTop-1 accuracy is the conventional accuracy, which means that the model '
    + 'answer \n(the one with the highest probability) must be exactly the expected answer.'
    +f'\n\n{top1_labels[0][0]}\n'
    + '\n\nTop-5 accuracy means that any of your model that gives 5 highest probability\n answers'
    + ' that must match the expected answer.\n'
    )

for i in range(len(top5_labels[0])):
  print(top5_labels[0][i])

"""Historical outputs:
---

Image 1: Lion.
```
> ('n02129165', 'lion', 0.9288668)
('n02125311', 'cougar', 0.00089567434)
('n02129604', 'tiger', 0.0007001101)
('n02130308', 'cheetah', 0.0003819853)
('n02106030', 'collie', 0.00034011155)
```
---

Image 2: Bike.
```
> ('n03792782', 'mountain_bike', 0.6611452)**
('n02835271', 'bicycle-built-for-two', 0.053169418)
('n04482393', 'tricycle', 0.008381028)
('n04235860', 'sleeping_bag', 0.0054462613)
('n03764736', 'milk_can', 0.0039183004)
```
---

Image 3: Coffee.
```
> ('n07930864', 'cup', 0.5445497)
('n07920052', 'espresso', 0.1077343)
('n03063599', 'coffee_mug', 0.095277004)
('n07932039', 'eggnog', 0.021933636)
('n03297495', 'espresso_maker', 0.021787835)
```
---

Image 4: Kite.
```
> ('n03355925', 'flagpole', 0.46088904)
('n03888257', 'parachute', 0.40852448)
('n03733131', 'maypole', 0.021915112)
('n03976657', 'pole', 0.006420887)
('n03944341', 'pinwheel', 0.004615334)
```
---

Image 5: Snail.
```
> ('n01944390', 'snail', 0.9255499)
('n01945685', 'slug', 0.002956227)
('n01943899', 'conch', 0.0021765176)
('n01986214', 'hermit_crab', 0.0005132853)
('n01968897', 'chambered_nautilus', 0.0004910154)
```
---

Image 6: Bench.
```
('n03891251', 'park_bench', 0.94239813)
('n02747177', 'ashcan', 0.0023240522)
('n02892201', 'brass', 0.0006186887)
('n09332890', 'lakeside', 0.00044254796)
('n03991062', 'pot', 0.00028989476)
```
---

Image 7: Leopard.
```
> ('n02128385', 'leopard', 0.92392915)
('n02128925', 'jaguar', 0.010171026)
('n02130308', 'cheetah', 0.0016932078)
('n02128757', 'snow_leopard', 0.0015446297)
('n02606052', 'rock_beauty', 0.0009375865)
```
---

Image 8: Cat.
```
> ('n02123394', 'Persian_cat', 0.8734948)
('n02127052', 'lynx', 0.006491119)
('n02328150', 'Angora', 0.004858171)
('n02123045', 'tabby', 0.0020751117)
('n03721384', 'marimba', 0.0013176007)
```
---

Image 9: Bus.
```
> ('n03769881', 'minibus', 0.7118722)
('n03977966', 'police_van', 0.12225373)
('n02701002', 'ambulance', 0.039510496)
('n03796401', 'moving_van', 0.009279929)
('n03770679', 'minivan', 0.0051392214)
```
---

Image 10: Hog.
```
> ('n02395406', 'hog', 0.87603027)
('n02396427', 'wild_boar', 0.041023444)
('n03935335', 'piggy_bank', 0.004426862)
('n02927161', 'butcher_shop', 0.0006680207)
('n02364673', 'guinea_pig', 0.000499933)
```

## (b) Use the loaded model to extract the features from each image. Print out the features and the shape of the extracted features.
"""

# Using a pre-trained model in Keras to extract the feature of a given image.
print(f'Image Size of {name_image}: {img.size}')
print(f'Shape of feature extraction: {predictions.shape}\n')
print(f'Features:\n{predictions}')

# Write the mean value of the prediction into the dictionary.
IMG_FEATURES[name_image] = np.ndarray.mean(predictions)

"""## (c) Based on the calculated features in the previous part, which two images are more similar to each other?

"""

# Prints the mean values of the images. It'll fill up as images are compiled.
for i, f in IMG_FEATURES.items():
  print("{} ({})".format(i, f))

"""---
Taken the mean value of every feature set to calculate how close each value is from another. These are the historical results.

```
Lion (0.001000000280328095)
Bike (0.0009999998146668077)
Coffee (0.0010000000474974513)
Kite (0.0010000000474974513)
Snail (0.000999999581836164)
Bench (0.0009999998146668077)
Leopard (0.0010000000474974513)
Cat (0.0009999996982514858)
Bus (0.0009999996982514858)
Pig (0.001000000280328095)
```
Following this approach, many images share same overal similarities. Having matching images like:
*   `Cat` and `Bus`.
*   `Bike` and `Bench`.
*   `Lion` and `Hog`.
*   `Kite` with `Coffee` and `Leopard`.

Making the `Snail` the only one with no shared similarities.
"""
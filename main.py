from keras.models import Model

import numpy as np

from keras_vggface.vggface import VGGFace
img_size = (224, 224)
vggface = VGGFace(model='vgg16')
feature_model = Model(vggface.input, vggface.get_layer('fc6').output)
print(feature_model.summary())
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras._tf_keras.keras.preprocessing import image
import cv2

model_path = 'cnn_model.h5'
cnn_model = tf.keras.models.load_model(model_path)

class_names = [
    'Jerawat dan Rosacea',
    'Karsinoma Sel Basal dan Lesi Maligna Lainnya',
    'Dermatitis Atopik',
    'Selulitis, Impetigo, dan Infeksi Bakteri Lainnya',
    'Eksim',
    'Eksantema dan Erupsi Obat',
    'Herpes, HPV, dan Infeksi Menular Seksual Lainnya',
    'Penyakit dan Gangguan Pigmentasi',
    'Lupus dan Penyakit Jaringan Ikat Lainnya',
    'Melanoma, Kanker Kulit, Nevus, dan Tahi Lalat',
    'Dermatitis Kontak (Poison Ivy)',
    'Psoriasis, Liken Planus, dan Penyakit Terkait',
    'Keratosis Seboreik dan Tumor Jinak Lainnya',
    'Penyakit Sistemik',
    'Tinea, Ringworm, Kandidiasis, dan Infeksi Jamur Lainnya',
    'Urtikaria (Biduran)',
    'Tumor Vaskular',
    'Vaskulitis',
    'Kutil, Molluscum, dan Infeksi Virus Lainnya'
]

images= []
img_path = 'jerawatan.jpg'
img = image.load_img(img_path, target_size=(192, 192,3))
img_size = (192, 192, 3)
img_array = np.asarray(cv2.resize(cv2.imread(img_path, cv2.IMREAD_COLOR), img_size[0:2])[:, :, ::-1])
images.append(img_array)
images = np.asarray(images)


predictions = cnn_model.predict(images, verbose=0)[0]

plt.figure(figsize=(10, 5))
plt.imshow(img)
plt.axis('off')
plt.title(class_names[np.argmax(predictions)] + ' - ' + str(round(predictions[np.argmax(predictions)], 2)))
plt.show()

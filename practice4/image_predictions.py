import numpy as np
from keras.applications import VGG19, InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

models = {
    'VGG16': VGG16,
    'VGG19': VGG19,
    'InceptionV3': InceptionV3,
    'MobileNet': MobileNet,
    'ResNet50': ResNet50
}

img_path = 'images/fruits.jpg'

for key, model_type in models.items():
    model = model_type()

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = model.predict(x)

    # decode the results into a list of tuples (class, description, probability)
    decoded_predictions = decode_predictions(predictions, top=5)

    print('Prediction by {} model:\n'.format(key))

    for i in range(0, len(decoded_predictions[0])):
        # retrieve the most likely result, e.g. highest probability
        labelTemp = decoded_predictions[0][i]
        # print the classification: the image class and probability
        print('%s (%.2f%%);' % (labelTemp[1], labelTemp[2] * 100))

    print('\n')

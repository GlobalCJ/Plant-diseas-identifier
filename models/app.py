import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from keras.preprocessing import image

model = tf.keras.models.load_model('apple.h5')

st.write("""
         # Plant Disease Prediction
         """
         )
st.write("""This is a simple image classification web app to predict plant diseases and give you some tips on how to manage them.

- complex
- frog_eye_leaf_spot
- frog_eye_leaf_spot_complex
- healthy
- powederly_mildew
- powederly_mildew_complex
- rust
- rust_complex
- rust_frog_eye_leaf_spot
- scab
- scab_frog_eye_leaf_spot
- scab_frog_eye_leaf_spot_complex

Created by: Audrya Kerenhappukh, Anuruddha Kristhombuge and Cihan Acilov""")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

def load_image(image_data):

        size = (224,224)

        img = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img_tensor = image.img_to_array(img)                    # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
                                    # imshow expects values in the range [0, 1]

        return img_tensor

if file is None:
    st.text("Please upload an image file")
else:
    # Get the image
    file_img = Image.open(file)
    st.image(file_img, use_column_width=True)

    # Pre proces it
    img = load_image(file_img)

    # Get the prediction with highest confidence
    pred = model.predict(img)
    class_pred = np.argmax(pred, axis = 1)

    # Collect the possible classes
    plant_classes = ['complex', 'frog_eye_leaf_spot', 'frog_eye_leaf_spot_complex',
                 'healthy','powederly_mildew',
                 'powederly_mildew_complex', 'rust',
                 'rust_complex', 'rust_frog_eye_leaf_spot',
                 'scab', 'scab_frog_eye_leaf_spot', 'scab_frog_eye_leaf_spot_complex']




    output = plant_classes[int(class_pred)]
    # Write output to screen
    st.write(f"this plant is classified as :{output}")

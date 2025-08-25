import tensorflow as tf
from PIL import Image
import numpy as np
import io

def predict_disease(model, image):
    predictions = model.predict(image / 255.0)
    return np.argmax(predictions, axis=1)

disease_statuses = ["Healthy", "Multiple Diseases", "Rust", "Scab"]

def generate_results(predictions_arr):
    class_index = int(predictions_arr[0])
    status = disease_statuses[class_index]

    result = {
        "status": f"has {status}" if status != "Healthy" else f"is {status}",
    }
    return result

def preprocess_image(image):
    image = np.array(image)
    image_pil = Image.fromarray(image)
    image_resized = image_pil.resize((512, 512), Image.LANCZOS)
    image_array = np.array(image_resized)
    image_processed = image_array[np.newaxis, :, :, :3]
    return image_processed


def load_trained_model(path='./model/model.h5'):
    xception_model = tf.keras.models.Sequential([
        tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    inputs = tf.keras.Input(shape=(512, 512, 3))
    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)
    outputs = tf.keras.layers.average([densenet_output, xception_output])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.load_weights(path)
    return model

def main():
    model = load_trained_model()
    image_path = input("Please enter the path to your mulberry plant leaf image (e.g., './leaf_image.jpg'): ").strip()

    try:
        with open(image_path, 'rb') as img_file:
            image = Image.open(io.BytesIO(img_file.read()))
    except FileNotFoundError:
        print(f"File not found at {image_path}. Please check the path and try again.")
        return
    except IOError:
        print(f"Error loading image. Please check if the file is a valid image.")
        return

    image_array = preprocess_image(image)
    predictions_arr = predict_disease(model, image_array)
    result = generate_results(predictions_arr)

    print(f"\nPrediction Results:")
    print(f"The plant {result['status']}.")

if __name__ == "__main__":
    main()


# Plant Disease Detection using Deep Learning

This project aims to detect and classify plant diseases from images of plant leaves using a machine learning model. It utilizes a combination of transfer learning models (Xception and DenseNet121) for robust predictions. 

## Features
- **Image Preprocessing**: Automatically resizes and processes images to be compatible with the model.
- **Disease Classification**: Identifies four classes of plant health:
  - Healthy
  - Multiple Diseases
  - Rust
  - Scab
- **Prediction Scores**: Outputs the probability score of the predicted class for confidence assessment.

---

## How It Works

1. **Input Image**: Users provide an image of a plant leaf.
2. **Preprocessing**: The image is resized to 512x512 pixels and normalized.
3. **Prediction**: The pre-trained model predicts the class of the disease and its confidence score.
4. **Result Display**: Outputs the health status and prediction accuracy.

---

## Installation

### Prerequisites
1. Python 3.8 or above
2. Virtual environment (optional but recommended)

### Required Libraries
Install the dependencies using `pip`:

```bash
pip install tensorflow pillow numpy
```

---

## Project Files
- **`main.py`**: Main script to run the prediction pipeline.
- **`models/model.h5`**: Pre-trained model weights (download link provided below).
- **Sample Images**: Use your own plant leaf images for testing.

---

## Usage

1. **Download Model Weights**:
   Download the pre-trained model weights from the link below and place them in the `./models/` directory:

   [Download Model Weights](https://drive.google.com/file/d/1bTie0IcrtEp_SB489E0uUmGwpjGkdCI7/view?usp=sharing)

2. **Run the Script**:
   Execute the script with:
   ```bash
   python main.py
   ```

3. **Provide Input**:
   Enter the path to your image file (e.g., `./leaf_image.jpg`) when prompted.

4. **View Results**:
   The script will output the disease status and confidence score.

---

## Model Details
This project combines **Xception** and **DenseNet121** transfer learning models. The final prediction is the average output of both models, enhancing reliability.

### Model Architecture
- **Base Models**: Xception and DenseNet121, pre-trained on ImageNet.
- **Final Layer**: A softmax classifier with four output nodes corresponding to the plant statuses.

---

## Example Output

```plaintext
Please enter the path to your plant leaf image (e.g., './leaf_image.jpg'): ./sample_leaf.jpg

Prediction Results:
The plant has Rust with a prediction score of 85%.
```

---

## Contributions
Feel free to contribute by:
- Adding more disease classes.
- Improving the model architecture or preprocessing pipeline.
- Optimizing prediction speed.

Fork this repository and submit a pull request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

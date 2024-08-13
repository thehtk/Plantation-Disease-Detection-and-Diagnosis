# Plantation Disease Detection and Diagnosis

## Overview

Plantation Disease Detection and Diagnosis is a deep learning-based project designed to automatically detect and diagnose plant diseases. Leveraging the power of neural networks, this system aims to enhance agricultural plant protection by providing accurate and efficient disease recognition, which traditionally relies on manual processes.

## Features

- **Deep Learning Model:** Utilizes a neural network to learn and extract features for plant disease recognition.
- **Automated Feature Selection:** Eliminates the need for manual selection of disease attributes, providing a more objective approach.
- **Increased Productivity:** Accelerates research in agricultural technology by automating disease feature extraction and recognition.

## Project Details

Deep learning, a branch of Artificial Intelligence, has gained significant attention due to its ability to automatically learn and extract features. This project applies deep learning to the agricultural domain, specifically for plant disease recognition and diagnosis. Traditional methods of disease detection involve manual selection of disease marks, which can be subjective and time-consuming. Our system mitigates these drawbacks by employing a neural network that objectively processes disease features, enhancing research productivity and technological advancements.

## Technologies Used

- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow/Keras (or any other framework you used)
- **Data Processing:** NumPy, Pandas
- **Image Processing:** OpenCV, PIL

## How It Works

1. **Data Collection:** The system is trained on a dataset of plant images, each labeled with the corresponding disease.
2. **Preprocessing:** Images are processed and augmented to enhance the model's performance.
3. **Model Training:** A deep learning model is trained using the processed data to learn the features of various plant diseases.
4. **Prediction:** The trained model is used to predict the disease in new plant images.
5. **Diagnosis:** The system provides a diagnosis based on the predicted disease, offering insights for further actions.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/thehtk/Plantation-Disease-Detection-and-Diagnosis.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the application:
    ```bash
    python main.py
    ```

## Usage

1. **Train the Model:** If you have a new dataset, you can train the model by running:
    ```bash
    python train.py --dataset path_to_dataset
    ```
2. **Predict Disease:** To predict disease from a new plant image:
    ```bash
    python predict.py --image path_to_image
    ```

## Results

The project has demonstrated promising results in accurately detecting and diagnosing plant diseases, with potential applications in real-world agricultural settings.

## Publication

This project is based on the research paper published by Hitesh Kumar, Kunal, Shreya Jadon, Saksham Sharma , Mrs. Upasana Tiwari. The paper provides a detailed analysis of the methodologies used and the results obtained. [Read Paper](https://www.trendytechjournals.com/ijtret/volume8/issue1-3.pdf)

## Contributing

If you wish to contribute to this project, feel free to submit a pull request or open an issue for any bugs or feature requests.

## Acknowledgements

- Thanks to the community for providing datasets and tools that made this project possible.
- Special thanks to Kunal, Shreya Jadon, Saksham Sharma , Mrs. Upasana Tiwari.

## Contact

For more information, feel free to contact me at [hitesh70001@gmail.com].

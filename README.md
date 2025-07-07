# EcoQCode Detector

This project is a web-based application that uses machine learning models to detect and classify "EcoQCodes" in real-time from a video stream. It leverages ONNX Runtime for in-browser inference, allowing for client-side processing without needing a powerful backend server.

## Features

- **Real-time Object Detection**: Detects EcoQCodes from a live camera feed using a YOLO-based model.
- **EcoQCode Classification (OCR)**: Classifies a single captured image to determine if it is an EcoQCode.
- **In-browser ML**: Uses ONNX Runtime to run the detection and classification models directly in the browser.
- **Responsive UI**: A simple and clean interface that displays the camera feed and results.
- **Detection Banner**: A banner appears when EcoQCodes are detected or classified.

## How It Works

1.  **Camera Access**: The application requests access to the user's camera via the browser.
2.  **Model Loading**: Depending on the module used, either the YOLO object detection model (`model_simplified.onnx`) or the OCR classification model (`ecoq_classifier.onnx`) is loaded into the browser using `onnxruntime-web`.
3.  **Frame Processing**: Video frames are captured, preprocessed according to the specific model's requirements (e.g., resizing, normalization).
4.  **Inference**: The loaded model performs inference on the processed frame to either detect EcoQCodes (YOLO) or classify a single EcoQCode (OCR).
5.  **Post-processing & Visualization**: The model's output is post-processed. For detection, bounding boxes are drawn. For classification, a score or label is provided. Results are displayed on a canvas overlaying the video feed.

## Setup and Usage

To run this project locally, you need Python installed.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd web-ecoqcode
    ```

2.  **Start the server:**
    A simple Flask server is provided to serve the files.
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate

    pip install Flask
    python app.py
    ```
    Alternatively, you can use Python's built-in HTTP server:
    ```bash
    python -m http.server 8000
    ```

3.  **Access the application:**
    Open your web browser and navigate to `http://localhost:8000`. You may need to allow camera access for the site.
    To access the YOLO detection module, navigate to `http://localhost:8000/yolo/index.html`.
    To access the OCR classification module, navigate to `http://localhost:8000/OCR/index.html`.

4.  **Testing on a mobile device (e.g., iPhone):**
    To run the inference on a mobile device, you need to serve the application over HTTPS, as browsers require a secure context to access the camera. `ngrok` is a good tool for this.

    First, start the local server as described above:
    ```bash
    python -m http.server 8000
    ```

    Then, in a new terminal, use `ngrok` to create a public HTTPS tunnel to your local server:
    ```bash
    ngrok http 8000
    ```
    `ngrok` will provide you with a public HTTPS URL. Open this URL on your mobile device to access the application and use the camera for real-time detection/classification.

## File Descriptions

- **`README.md`**: This file, providing an overview of the project.
- **`index.html`**: The main HTML file for the root directory (likely a placeholder or landing page).
- **`style.css`**: Contains general styles for the application's UI.
- **`app.py`**: A simple Flask web server to serve the static files.
- **`ecoqcode.jpg`**: A sample image of an EcoQCode.

### YOLO Module Files (`yolo/`)

- **`yolo/index.html`**: The HTML file for the YOLO object detection module.
- **`yolo/app.js`**: The core JavaScript file for the YOLO module, handling camera access, model inference, and rendering of detection bounding boxes.
- **`yolo/style.css`**: Styles specific to the YOLO module.
- **`yolo/model/model_simplified.onnx`**: The pre-trained ONNX model for EcoQCode object detection.

### OCR Module Files (`OCR/`)

- **`OCR/index.html`**: The HTML file for the OCR classification module.
- **`OCR/app.js`**: The core JavaScript file for the OCR module, handling camera access, image preprocessing, and classification using the ONNX model.
- **`OCR/style.css`**: Styles specific to the OCR module.
- **`OCR/model/ecoq_classifier.onnx`**: The pre-trained ONNX model for EcoQCode classification.

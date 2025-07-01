# EcoQCode Detector

This project is a web-based application that uses a machine learning model to detect "EcoQCodes" in real-time from a video stream. It leverages ONNX Runtime for in-browser inference, allowing for client-side object detection without needing a powerful backend server.

## Features

- **Real-time Detection**: Detects EcoQCodes from a live camera feed.
- **In-browser ML**: Uses ONNX Runtime to run the detection model directly in the browser.
- **Responsive UI**: A simple and clean interface that displays the camera feed and detection results.
- **Detection Banner**: A banner appears when 3 or more EcoQCodes are detected.

## How It Works

1.  **Camera Access**: The application requests access to the user's camera via the browser.
2.  **Model Loading**: An ONNX model (`model_simplified.onnx`) is loaded into the browser using `onnxruntime-web`.
3.  **Frame Processing**: Video frames are captured, preprocessed, and fed into the ONNX model.
4.  **Inference**: The model performs inference on the frame to detect EcoQCodes.
5.  **Post-processing & Visualization**: The model's output is post-processed to get bounding box coordinates and scores. A mask and boxes are drawn on a canvas overlaying the video feed to show the detections.

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

## File Descriptions

- **`index.html`**: The main HTML file that structures the web page.
- **`style.css`**: Contains the styles for the application's UI.
- **`app.js`**: The core JavaScript file that handles camera access, model inference, and rendering.
- **`app.py`**: A simple Flask web server to serve the static files.
- **`model/model_simplified.onnx`**: The pre-trained ONNX model for EcoQCode detection.
- **`ecoqcode.jpg`**: The logo image for the project.

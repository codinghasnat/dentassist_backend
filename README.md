# DentAssist Backend

This is the backend service for the DentAssist application, responsible for AI-powered dental analysis from X-ray images.

## Features

- YOLO-based tooth detection
- Binary classification to filter non-tooth objects
- Disease classification for detected teeth
- PDF report generation with oral health scoring and recommendations

## API Endpoints

### `/analyze` (POST)
- **Description**: Analyzes an X-ray image to detect teeth and classify diseases
- **Input**: Multipart form with an image file
- **Output**: JSON with original image, annotated image, and detected teeth details

### `/disease_classify` (POST)
- **Description**: Classifies disease in a single tooth image
- **Input**: Multipart form with an image file
- **Output**: JSON with disease classification and confidence

### `/generate_report` (POST)
- **Description**: Generates a comprehensive dental health PDF report
- **Input**: JSON with original image, annotated image, and teeth by disease
- **Output**: JSON with report ID and download URL

### `/download_report/<report_id>` (GET)
- **Description**: Downloads a previously generated PDF report
- **Input**: Report ID in URL path
- **Output**: PDF file

## Structure

- `app.py`: Main Flask application with API endpoints
- `detector.py`: YOLO-based tooth detection
- `binary_classifier.py`: Filters non-tooth objects
- `disease_classifier.py`: Classifies dental conditions
- `utils/`: Utility functions
  - `image_processing.py`: Image annotation and processing
  - `report_generator.py`: PDF report generation

## Installation

```bash
pip install -r requirements.txt
```

## Running the Service

```bash
python app.py
```

The server will start on http://127.0.0.1:5000

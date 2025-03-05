# Hand Gesture Recognition Web App
## Project Overview
This project aims to develop a web application that allows students to check their academic status using only touchless hand gestures. The application is specifically designed to improve accessibility for individuals with motor disabilities by leveraging machine learning techniques for gesture recognition.


## 🎥 Demo
https://github.com/user-attachments/assets/85c27091-7946-4c3a-a8f4-dd51524cc123

## Technologies Used
- **Machine Learning**: TensorFlow, scikit-learn
- **Computer Vision**: MediaPipe for hand tracking and gesture recognition
- **Web Development**: Flask (Python), HTML, CSS, JavaScript
- **Database**: MariaDB
- **Other Libraries**: OpenCV for real-time image processing

## Project Architecture
The system follows a **Client-Server Architecture** with the following components:

1. **Client-Side (Frontend):**
   - Captures images from a webcam
   - Sends frames to the server for processing
   - Displays the results (recognized gestures and UI interactions)
   
2. **Server-Side (Backend - Flask):**
   - Processes images using MediaPipe for hand landmark detection
   - Runs a trained neural network model to classify gestures
   - Interacts with the database to retrieve and store user data
   
3. **Database (MariaDB):**
   - Stores student information, grades, and academic records

## Installation & Setup
### Prerequisites
- Python 3.8+
- MySQL or MariaDB
- Virtual environment (optional but recommended)

### Steps to Set Up
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

4. Set up the database:
   - Create a MySQL database named `licenta`
   - Import the provided SQL file to initialize the tables:
     ```bash
     mysql -u root -p licenta < database.sql
     ```
5. Run the Flask application:
   ```bash
   python app.py
   ```
6. Access the application at `http://localhost:5000`

## Model Training & Testing
The neural network model is a **Multi-Layer Perceptron (MLP)** that classifies gestures based on hand landmarks. The model is trained using MediaPipe-extracted features and evaluated through:
- Loss and accuracy metrics
- ROC curves
- Precision, recall, and F1-score

### Performance Testing
Several configurations were tested by varying:
- Learning rate (0.0001, 0.001, 0.01)
- Epochs (250, 500)
- Batch size (64, 128)

Results showed stable convergence with high accuracy (~95-99%) in the best configurations.

## Future Improvements
- Expanding the dataset for better generalization
- Enhancing UI/UX for smoother interaction
- Adding more gesture-based commands for additional functionalities
- Exploring deep learning models (e.g., CNNs, LSTMs) for improved recognition accuracy
- Implementing security measures to protect user data

## License
This project is open-source and available under the MIT License.

## Contributors
- **Cozmina Scorobete**
- **Project Supervisor:** Dr. Mario Reja

---
For any questions or suggestions, feel free to open an issue or submit a pull request! 🚀



## 📄 License
This project is licensed under the **GNU General Public License**.

---


## 📜 GNU General Public License (GPL)

```
GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007

Copyright (c) 2024 Cozmina Scorobete

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```


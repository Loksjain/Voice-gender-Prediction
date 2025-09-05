# Voice Gender Prediction

This project uses machine learning to predict a person's gender (male or female) based on their voice's acoustic properties. It includes a Jupyter Notebook for data analysis and model training, and a Django web application to serve the trained model.

## Features

- **Data Analysis:** Exploratory Data Analysis (EDA) and visualization of voice features.
- **Machine Learning Model:** Compares Support Vector Classifier (SVC) and Random Forest Classifier, ultimately selecting the latter for its high accuracy.
- **Web Interface:** A Django-based web application to predict gender from input voice features.
- **Deployment Ready:** Configured for deployment on platforms like Heroku.

## Dataset

The project utilizes the "Voice Gender" dataset, which was created to identify a voice as male or female based on acoustic properties of the voice and speech.

- **Source:** The dataset is based on the work of Kory Becker and is available on [Kaggle](https://www.kaggle.com/primaryobjects/voicegender).
- **Content:** It contains 3,168 recorded voice samples from male and female speakers. The dataset consists of 20 acoustic features extracted from the audio samples and a "label" column indicating the gender.

The features include:
`meanfreq`, `sd`, `median`, `Q25`, `Q75`, `IQR`, `skew`, `kurt`, `sp.ent`, `sfm`, `mode`, `centroid`, `meanfun`, `minfun`, `maxfun`, `meandom`, `mindom`, `maxdom`, `dfrange`, `modindx`.

## Methodology

The `GenderPrediction.ipynb` notebook details the following steps:

1.  **Data Loading and Exploration:** The dataset is loaded, and initial analysis is performed to check for null values and understand data distribution.
2.  **Visualization:**
    - A correlation heatmap is generated to understand the relationships between different features.
    - Histograms are plotted to visualize the distribution of each feature for male and female labels.
3.  **Feature Selection:** Based on high correlation and feature distribution analysis, several redundant or less informative columns (`dfrange`, `kurt`, `sfm`, `meandom`, `meanfreq`) were dropped to improve model performance and reduce complexity.
4.  **Model Training and Evaluation:**
    - The data is split into training (80%) and testing (20%) sets.
    - Two models are trained and evaluated:
        - **Support Vector Classifier (SVC):** Achieved ~73% accuracy on the test set.
        - **Random Forest Classifier:** Achieved **98% accuracy** on the test set.
5.  **Model Selection and Saving:** Due to its superior performance, the Random Forest model was chosen and saved as `voice_model.pickle` using pickle for use in the web application.

## Technology Stack

- **Language:** Python 3
- **Data Analysis & ML:** Jupyter Notebook, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, mglearn
- **Web Framework:** Django
- **Deployment:** Gunicorn

## Directory Structure

```
.
├── GenderPrediction.ipynb      # Notebook for data analysis and model building.
├── gender_voice_dataset.csv    # The dataset.
├── mysite/                     # Django project directory.
│   ├── manage.py
│   ├── mysite/                 # Django app configuration.
│   ├── polls/                  # Django app for handling predictions.
│   ├── templates/              # HTML templates for the web interface.
│   └── voice_model.pickle      # The saved Random Forest model.
├── Procfile                    # For Heroku deployment.
├── requirements.txt            # Project dependencies.
└── README.md                   # This file.
```

## Setup and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Loksjain/Voice-Gender.git
cd Voice-Gender-Prediction-code
```

### 2. Machine Learning Model (Jupyter Notebook)

To explore the data analysis and model building process, you can run the Jupyter Notebook. Note that the notebook was originally run in a Google Colab environment, so you may need to adjust file paths for `gender_voice_dataset.csv` and the saved `voice_model.pickle` file.

1.  **Install dependencies:**
    ```bash
    pip install jupyter pandas numpy scikit-learn matplotlib seaborn mglearn
    ```
2.  **Run Jupyter:**
    ```bash
    jupyter notebook GenderPrediction.ipynb
    ```

### 3. Django Web Application

The project includes a Django web application to serve the model.

1.  **Create a Virtual Environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    Navigate to the Django project directory and run the server.
    ```bash
    cd mysite
    python manage.py migrate
    python manage.py runserver
    ```
    The application will be available at `http://127.0.0.1:8000/`.

## Results

The Random Forest Classifier model achieved an accuracy of **98%** on the test set, demonstrating high effectiveness in predicting gender from voice features.

## Deployment

This project is configured for deployment on Heroku. The `Procfile` specifies the Gunicorn web server to run the WSGI application.

## Security Note

The `SECRET_KEY` is currently hardcoded in `mysite/mysite/settings.py`. For a production environment, it is strongly recommended to move this to an environment variable for better security.

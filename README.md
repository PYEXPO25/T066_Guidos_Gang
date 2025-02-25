# T066_Guidos_Gang
![PyExpo Logo](media/pyexpo-logo.png)

---

## Problem Statement

*Problem Statement ID – PY058

IPL 2025 Winner Prediction
        The challenge entails creating a predictive model that leverages machine learning techniques in Python to forecast the outcome of the IPL 2025 tournament. To achieve this, historical datasets comprising match results, player performances, team statistics, venue details, and other relevant information will be utilized. The aim is to build a robust prediction system that can analyze past trends, player form, team strategies, and various influencing factors to provide accurate predictions about the potential winner of the IPL 2025 season

---

## Overview

Machine learning prediction using scikit learn (random forest and linear regression) additionally AI suggestions on the venues and combinations of the teams . Training the model with past records , Will predict even in the critical condition like key players early wickets and players dependency analysis by balancing and Recalculating the result .


---

## Team Members

*Team ID – T066

List your team members along with their roles.

- *RAAHUL KANAA.K* - Team Leader
- *VINOTHAA.S.P* - BACKEND
- *PRAVIN RAJA.M* - FRONTEND
- *SAMASTHUTHI.P* - FRONTEND
- *RAGUL.S* - FRONTEND
- *SRIRAM.S* - FRONTEND
- 
![team_photo](media/team-photo.jpg)

---

## Technical Stack

List the technologies and tools used in the project. For example:

- *Frontend:* HTML, CSS, JavaScript
- *Backend:* Django
- *Database:* MySQL
- *Other Tools:*  Git

---

## Getting Started

Follow these steps to clone and run the application locally.

### Prerequisites

1. Install [Python](https://www.python.org/downloads/).
2. Install [Git](https://git-scm.com/).
3. Clone this repository:
   bash 
   git clone https://github.com/PYEXPO25/T066_Guidos_Gang.git
   


---

# IPL 2025 Winner Prediction Installation Guide

This guide outlines how to set up the project—which uses Django for the back end, MySQL for the database, and Scikit-Learn for machine learning predictions—on your local machine.

## Prerequisites

- **Python 3.x** installed  
- **MySQL** installed and running  
- **Git** (if cloning the repository)  
- Basic familiarity with the command line

## Step-by-Step Installation

### 1. Clone the Repository

If you haven’t already downloaded the project, clone the repository:

```bash
git clone https://github.com/PYEXPO25/T066_Guidos_Gang.git
cd repository-name
```

### 2. Create a Virtual Environment

Create a dedicated Python virtual environment:

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

- **Windows:**

  ```bash
  venv\Scripts\activate
  ```

- **macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies

Install all required Python packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

*This should install Django, Scikit-Learn, MySQL connector libraries, and any other dependencies required by your project.*

### 5. Set Up the MySQL Database

1. **Create a New Database:**  
   Log in to MySQL and create a new database (e.g., `ipl2025_db`):

   ```sql
   CREATE DATABASE ipl2025_db;
   ```

2. **Configure Database Settings:**  
   Open your Django project’s `settings.py` file and update the `DATABASES` section with your MySQL credentials:

   ```python
   DATABASES = {
       'default': {
           'ENGINE': 'django.db.backends.mysql',
           'NAME': 'ipl2025_db',
           'USER': 'your_mysql_user',
           'PASSWORD': 'your_mysql_password',
           'HOST': 'localhost',
           'PORT': '3306',
       }
   }
   ```

### 6. Apply Django Migrations

Run Django’s migration commands to set up the database schema:

```bash
python manage.py makemigrations
python manage.py migrate
```

### 7. Training the Machine Learning Model

If your project includes a separate script for training the ML model (using Random Forest and Linear Regression):

```Give command 
      python manage.py shell

   then run ,
      exec(open("C:/Users/raahu/Desktop/Django8/mywebsite/pages/train.py").read())

   Output ,
      Model saved successfully at: C:/Users/raahu/Desktop/Django8/mywebsite/pages/prediction/ml_model.pkl

   same for train1.py 
   

*Make sure the pkl file exists in your project directory if applicable.*

### 8. Run the Development Server

Start the Django development server:

```
python manage.py runserver
```

Then open your browser and navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) to view the application.





*Ensure that your training data is available in the expected location.*

### 

- **Environment Variables:**  
  You might need to set environment variables (like API keys) either in a `.env` file or directly in your hosting environment.

- **Deactivating the Virtual Environment:**  
  When you’re done working on the project, simply run:

  ```
  deactivate


---

These steps should help you get the project up and running locally. If any part of the installation differs (for example, file names or additional configuration files), adjust the instructions accordingly. Enjoy building and experimenting with your IPL 2025 Winner Prediction project!

## Start the Application

1. Run the Django application:
   python manage.py runserver

   
2. Open your browser and navigate to:
   
   (http://127.0.0.1:8000/)
   

---

## UI Overview

Images to demonstrate the user interface:

*Example pages:*

1. *Landing Page:*
   ![Landing Page Mockup](media/LoadingPage.png)

2. *Dashboard:*
   ![Dashboard Mockup](media/DashBoard.png)

3. *Analytics View:*
   ![Analytics Mockup](media/Analytics.png)

---

## Resources

### 📄 PowerPoint Presentation
[Click here to view the file](resource/)


### 🎥 Project Video
[Click here to view the project demo video](insert-drive-link-here)

### 📹 YouTube Link
[Watch the project on YouTube](insert-youtube-link-here)

---

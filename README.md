# AI-Powered Cooking Assistant

This repository contains the code and resources for the AI-Powered Cooking Assistant project. The goal of this project is to provide users with personalized recipe recommendations and cooking assistance through an interactive chatbot interface.

## Project Overview

### 1. Dataset Cleaning and Preparation
The initial dataset was highly disorganized, containing various types and unstructured data. The key steps in the cleaning process included:
- **Nutritional Data**: The nutritional information was initially in a variable format. This was split into distinct columns for easier access and analysis.
- **Ingredients List**: The ingredients were provided in a disordered string format. This list was cleaned and converted into a structured, usable format.

### 2. Exploratory Data Analysis (EDA) and Model Training
After cleaning, the data was used for EDA and training different models:
- **Algorithms Tried**: Various algorithms such as K-Nearest Neighbors (KNN), DBSCAN, and K-Means were evaluated.
- **Final Model**: Due to the dataset's unique structure, where each row represents a distinct recipe, KNN was found to be the most effective and was selected as the final model.

### 3. LLM Model Dataset and Fine-Tuning
A separate dataset (`data_for_llm`) was created to train a Large Language Model (LLM). This dataset included detailed cooking directions.
- **LLM Models Attempted**: I experimented with models like GPT-2 and T5-small, fine-tuning them using the `data_for_llm` dataset.
- **Challenges Faced**: Both models struggled with generating coherent cooking directions, often repeating sentences.
- **PDF Reading Model**: Additionally, a PDF reading model was developed using Mistral, which provided satisfactory results.

### 4. Continuous Chatbot Development
A continuous chatbot was developed using the Gemini Pro API. This chatbot allows users to interact seamlessly, providing cooking instructions and assistance in a continuous conversation format.

### 5. Streamlit Application
A Streamlit app was created to provide recipe recommendations based on user inputs:
- **Features**: The app includes fields for nutritional information, preparation time, and ingredient lists.
- **User Interface**: The recipe recommendation and chatbot functionalities are separated into two distinct pages, ensuring a user-friendly experience.

## How to Run the Project

### Prerequisites
- Python 3.7+
- Streamlit
- Transformers
- scikit-learn
- Other dependencies can be found in `requirements.txt`

### Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/muhammedap1//smartspoon.git
cd smartspoon
pip install -r requirements.txt


## Future works
- Improving LLM model performance for cooking directions.
- Integrating more dietary and allergy considerations into the recommendation system.
- Enhancing the chatbot's contextual understanding.


## License

This project is licensed under the MIT License.

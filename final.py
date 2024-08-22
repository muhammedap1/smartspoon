import pandas as pd
import numpy as np
import os
import google.generativeai as genai
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Load your datasets
df = pd.read_csv('data_for_llm.csv')
recipe_data = pd.read_csv("final_dataset.csv")

# Function to preprocess recipe data
def preprocess_data(data):
    vectorizer = TfidfVectorizer(max_features=2000)  # Reduced number of features
    X_ingredients = vectorizer.fit_transform(data['ingredients_list'])

    svd = TruncatedSVD(n_components=100)  # Reduced number of components
    X_ingredients_reduced = svd.fit_transform(X_ingredients)

    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(data[['prep_time', 'calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium', 'fiber']])

    X_combined = np.hstack([X_numerical, X_ingredients_reduced])

    knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
    knn.fit(X_combined)

    return vectorizer, svd, scaler, knn

def recommend_recipes(input_features, vectorizer, svd, scaler, knn, data):
    input_features_scaled = scaler.transform([input_features[:8]])  # Adjusted to include prep_time
    input_ingredients_transformed = vectorizer.transform([input_features[8]])
    input_ingredients_reduced = svd.transform(input_ingredients_transformed)
    input_combined = np.hstack([input_features_scaled, input_ingredients_reduced])
    distances, indices = knn.kneighbors(input_combined)
    recommendations = data.iloc[indices[0]]
    return recommendations[['recipe_name', 'prep_time', 'ingredients_list']].head(5)

# Initialize the chatbot
def initialize_chatbot():
    genai.configure(api_key=os.getenv('google_api_key'))
    model = genai.GenerativeModel('gemini-pro')
    
    if 'chat' not in st.session_state:
        st.session_state['chat'] = model.start_chat(history=[])
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def enhance_cooking_directions(directions):
    """Use the model to enhance the grammar and context of cooking directions."""
    prompt = f"Please rephrase and correct the following cooking directions to make them more clear and grammatically correct:\n\n{directions}"
    chat = st.session_state['chat']

    try:
        response = chat.send_message(prompt, stream=True)
        response.resolve()

        enhanced_directions = ""
        for candidate in response.candidates:
            for part in candidate.content.parts:
                enhanced_directions += part.text

        return enhanced_directions
    except genai.generative_models.BrokenResponseError:
        # Handle the broken response
        last_send, last_received = chat.rewind()
        return "I'm sorry, there was an issue processing the cooking directions. Please try again."

def get_gemini_response(question):
    # Check if the question is about a specific recipe
    for index, row in df.iterrows():
        if row['recipe_name'].lower() in question.lower():
            # Enhance the cooking directions using the model
            enhanced_directions = enhance_cooking_directions(row['formatted_directions'])
            return f"Here are the improved cooking directions for {row['recipe_name']}:\n\n{enhanced_directions}"

    # If no matching recipe, use generative model for a general response
    prompt = """
    You are an interactive recipe assistant. When a user enters the name of a recipe or a related topic, 
    first determine if there are multiple types or variations of the dish. For example, if the user asks for "biriyani," 
    ask them which type of biriyani they want, such as "chicken," "beef," or "vegetable." Similarly, if they ask for "cake," 
    ask them which type of cake they are interested in, such as "chocolate," "vanilla," or "red velvet."

    Once the type of dish is clarified, provide a list of the cooking vessels and ingredients needed for the recipe. 
    After listing these, ask the user, 'Would you like to start?' 
    If the user agrees to start, provide the first step or action needed. 
    After each step, ask the user if they would like to continue to the next step, 
    and guide them through the recipe step-by-step until they reach the end.

    If the user's request is not clear or does not match any recipe in the dataset, ask for more details to clarify their request.
    """

    full_prompt = f"{prompt}\n{question}"
    chat = st.session_state['chat']

    try:
        response = chat.send_message(full_prompt, stream=True)
        response.resolve()

        full_response_text = ""
        for candidate in response.candidates:
            for part in candidate.content.parts:
                full_response_text += part.text

        return full_response_text
    except genai.generative_models.BrokenResponseError:
        # Handle the broken response
        last_send, last_received = chat.rewind()
        return "I'm sorry, there was an issue with your request. Please try again."

def main():
    st.set_page_config(page_title="SmartSpoon🥄🍳👩")
    st.markdown("<h1 style='text-align: center; font-size: 48px;'>SmartSpoon🥄🍳👩</h1>", unsafe_allow_html=True)

    # Initialize chatbot
    initialize_chatbot()

    # Load and preprocess recipe data
    vectorizer, svd, scaler, knn = preprocess_data(recipe_data)

    # Sidebar for navigation
    st.sidebar.title("Pages")
    page = st.sidebar.radio("Select", ["Recipe", "Cook With ME"])

    if page == "Recipe":
        st.markdown("<p style='font-size: 16px; color: #555; font-weight: bold;'>Enter the preparation time, nutritional information, and ingredients to get recipe recommendations:</p>", unsafe_allow_html=True)
        prep_time = st.number_input('Preparation Time (minutes)', min_value=0.0, format="%.0f")
        calories = st.number_input('Calories', min_value=0.0, format="%.0f")
        fat = st.number_input('Fat', min_value=0.0, format="%.0f")
        carbohydrates = st.number_input('Carbohydrates', min_value=0.0, format="%.0f")
        protein = st.number_input('Protein', min_value=0.0, format="%.0f")
        cholesterol = st.number_input('Cholesterol', min_value=0.0, format="%.0f")
        sodium = st.number_input('Sodium', min_value=0.0, format="%.0f")
        fiber = st.number_input('Fiber', min_value=0.0, format="%.0f")
        ingredients = st.text_area('Ingredients (comma-separated)')

        if st.button('Get Recommendations'):
            input_features = [prep_time, calories, fat, carbohydrates, protein, cholesterol, sodium, fiber, ingredients]
            recommendations = recommend_recipes(input_features, vectorizer, svd, scaler, knn, recipe_data)
            
            st.write("### Recommended Recipes:")
            for idx, row in recommendations.iterrows():
                st.write(f"**{row['recipe_name']}**")
                st.write(f"Preparation Time: {row['prep_time']} minutes")
                st.write(f"Ingredients: {row['ingredients_list']}")
                st.write("---")

    elif page == "Cook With ME":
        st.markdown("<p style='font-size: 18px; font-weight: bold;'>Ask About Your recipe Here.</p>", unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.messages:
            st.chat_message(message['role']).markdown(message['content'])

        prompt = st.chat_input('Pass your prompt here')

        if prompt:
            st.chat_message('user').markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            response_text = get_gemini_response(prompt)
            st.chat_message('assistant').markdown(response_text)
            st.session_state.messages.append({'role': 'assistant', 'content': response_text})

if __name__ == "__main__":
    main()

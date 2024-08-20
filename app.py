import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


from dotenv import load_dotenv
load_dotenv()

import os
import google.generativeai as genai
def main():
    st.set_page_config(page_title="SmartSpoon🥄🍳👩")

    st.markdown("<h1 style='text-align: center; font-size: 48px;'>SmartSpoon🥄🍳👩</h1>", unsafe_allow_html=True)


    genai.configure(api_key=os.getenv('google_api_key'))

    ### function to load gemini pro model and get  response

    model=genai.GenerativeModel('gemini-pro')

    if 'chat' not in st.session_state:
        st.session_state['chat'] = model.start_chat(history=[])

    def get_gemini_response(question):
        prompt = """You are an interactive recipe assistant. When a user enters the name of a recipe or a related topic, 
        first provide a list of the cooking vessels and ingredients needed for the recipe. 
        After listing these, ask the user, 'Would you like to start?' 
        If the user agrees to start, provide the first step or action needed. 
        After each step, ask the user if they would like to continue to the next step, 
        and guide them through the recipe step-by-step until they reach the end.
        
        """
        full_prompt = f"{prompt}\n{question}"
        chat = st.session_state['chat']
        response = chat.send_message(full_prompt, stream=True)
        return response

    # Caching the data loading process
    @st.cache_data
    def load_data():
        return pd.read_csv("final_dataset.csv")

    data = load_data()

    # Preprocess Ingredients with limited TF-IDF features
    @st.cache_data
    def preprocess_data():
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

    vectorizer, svd, scaler, knn = preprocess_data()




    # Function to recommend recipes
    def recommend_recipes(input_features):
        input_features_scaled = scaler.transform([input_features[:8]])  # Adjusted to include prep_time
        input_ingredients_transformed = vectorizer.transform([input_features[8]])
        input_ingredients_reduced = svd.transform(input_ingredients_transformed)
        input_combined = np.hstack([input_features_scaled, input_ingredients_reduced])
        distances, indices = knn.kneighbors(input_combined)
        recommendations = data.iloc[indices[0]]
        return recommendations[['recipe_name', 'prep_time', 'ingredients_list']].head(5)

    # Streamlit app
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
            recommendations = recommend_recipes(input_features)
            
            st.write("### Recommended Recipes:")
            for idx, row in recommendations.iterrows():
                st.write(f"**{row['recipe_name']}**")
                st.write(f"Preparation Time: {row['prep_time']} minutes")
                st.write(f"Ingredients: {row['ingredients_list']}")
                st.write("---")

    elif page == "Cook With ME":
        

        st.markdown("<p style='font-size: 18px; font-weight: bold;'>Ask About Your recipe Here.</p>", unsafe_allow_html=True)
        
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history']=[]
        
        user_input=st.text_input('Input:',key='input')
        submit=st.button("Ask")
        
        if submit and user_input :
            response=get_gemini_response(user_input)
            
            ## Add user query and response to session chat history
            st.session_state['chat_history'].append(('You',user_input))
            st.subheader('The Response is ')
            for chunk in response:
                st.write(chunk.text)
                st.session_state['chat_history'].append(('Bot',chunk.text))
        
        st.markdown("---")  
        
        st.markdown("***")
        
        
        st.markdown("<h2 style='color: #007bff;'>The Chat History </h2>", unsafe_allow_html=True)
        for role,text in st.session_state['chat_history']:
            st.write(f'{role}: {text}' )
if __name__ == "__main__":
    main()
        
    # def get_gemini_response(prompt):
    #     chat = model.start_chat(history=[])
    #     response = chat.send_message(prompt, stream=True)
    #     return response

    # if 'messages' not in st.session_state:
    #     st.session_state.messages = []

    # def generate_response():
    #     query = st.session_state.input
    #     st.session_state.messages.append({"role": "user", "content": query})

    #     response = get_gemini_response(query)
    #     for chunk in response:
    #         st.session_state.messages.append({"role": "assistant", "content": chunk.text})

    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.write(message["content"])

    # input = st.chat_input("Your message")
    # if input:
    #     st.session_state.input = input
    #     generate_response()
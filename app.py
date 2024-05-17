import streamlit as st
import joblib
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load the models and other components with error handling
try:
    vectorizer = joblib.load('vectorizer.h5')
    nb_model = joblib.load('nb_model.h5')
    tokenizer = joblib.load('tokenizer.h5')
    label_encoder = joblib.load('label_encoder.h5')
    rnn_model = load_model('rnn_model.h5')
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Streamlit interface
st.title('Sentiment Analysis Using Reviews')
text = st.text_area('Enter a review text:', 'I loved my clothes when I purchased them')

if st.button('Predict'):
    if text:
        try:
            # For MultinomialNB model
            text_nb = vectorizer.transform([text])
            pred_proba_nb = nb_model.predict_proba(text_nb)[0]
            pred_label_nb = nb_model.predict(text_nb)[0]

            # For SimpleRNN model
            text_seq = tokenizer.texts_to_sequences([text])
            text_pad = pad_sequences(text_seq, maxlen=100)
            pred_proba_rnn = rnn_model.predict(text_pad)[0]
            pred_class_rnn = np.argmax(pred_proba_rnn)
            pred_label_rnn = label_encoder.classes_[pred_class_rnn]

            # Display predictions
            st.write("### Predictions:")
            st.write("#### MultinomialNB:")
            st.write(f"Predicted sentiment: {label_encoder.inverse_transform([pred_label_nb])[0]}")
            st.write("Predicted sentiment probabilities:")
            for sentiment, proba in zip(label_encoder.classes_, pred_proba_nb):
                st.write(f"{sentiment}: {proba:.2f}")

            st.write("#### SimpleRNN:")
            st.write(f"Predicted sentiment: {pred_label_rnn}")
            st.write("Predicted sentiment probabilities:")
            for sentiment, proba in zip(label_encoder.classes_, pred_proba_rnn):
                st.write(f"{sentiment}: {proba:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.write("Please enter a review text.")

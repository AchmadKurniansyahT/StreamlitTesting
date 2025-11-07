import streamlit as st
import pickle
import numpy as np

st.title("TIPS PREDICTION MODEL : RANDOM FOREST REGRESSOR")
st.markdown("___")

if 'model' not in st.session_state :
    model = pickle.load(open('RF Reg Model.sav', 'rb'))
    st.session_state['model'] = model

total_bill = st.number_input("Masukan Total Bill")
size = st.number_input("Masukan Jumlah Orang Dalam 1 Meja", value = 3, min_value = 1, max_value = 10)
smoker = st.selectbox("Perokok?", ['Ya', 'Tidak'])

if st.button('Klik Untuk Prediksi Tips!') :
    data = np.array([total_bill, size]).reshape(1,-1)
    hasil = st.session_state['model'].predict(data)
    st.write(f"Berikut Ini Hasil Prediksi Tip : $ {hasil[0]:.2f}")
else :
    st.write("Masukan Pilihan Yang Benar")
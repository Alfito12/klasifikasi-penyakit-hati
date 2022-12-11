import pandas as pd
import streamlit as st
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


st.write("""<h1>Aplikasi Pendeteksi Penyakit Hati di India</h1>""",unsafe_allow_html=True)

beranda, description, dataset, preprocessing, modeling, implementation = st.tabs(["Home","Description", "Dataset","Prepocessing", "Modeling", "Implementation"])

with beranda:
    st.write(""" """)
    st.image('https://img.freepik.com/free-vector/realistic-human-internal-organs-anatomy-composition-with-isolated-image-liver-vector-illustration_1284-66282.jpg', use_column_width=False, width=500)
    
    st.write(""" """)

    st.write("""
    Penyakit Liver istilah yang digunakan untuk setiap gangguan pada liver atau hati sehingga menyebabkan organ ini tidak dapat berfungsi dengan baik. Penderita penyakit hati terus meningkat karena konsumsi alkohol berlebihan, menghirup gas berbahaya, asupan makanan yang terkontaminasi, acar dan obat-obatan.
    """)

with description:
    st.subheader("""Pengertian""")
    st.write("""
    Dataset ini merupakan data gejala-gejala penderita penyakit hati yang terdapat di website kaggle.com, Kumpulan data ini berisi catatan pasien hati dan catatan pasien non hati yang dikumpulkan dari Timur Laut Andhra Pradesh, India. Dataset ini terdiri dari 11 atribut yaitu Age, Gender, Total Bilirubin, Direct Bilirubin, Alkaline Phosphotase,Alamine Aminotransferase, Aspartate Aminotransferase, Total Proteins, Albumin, Albumin and Globulin Ratio.
    """)

    st.subheader("""Kegunaan Dataset""")
    st.write("""
    Dataset ini digunakan untuk membangun model pembelajaran mesin logistik yang memprediksi apakah seorang pasien sehat (pasien non-hati) atau sakit (pasien hati) berdasarkan beberapa fitur klinis dan demografis (atau variabel input).
    """)

    st.subheader("""Fitur""")
    st.markdown(
        """
        Fitur-fitur yang terdapat pada dataset:
        - Age umur
        - Gender jenis kelamin (0=Female, 1=Male)
        - Total Bilirubin Pemeriksaan yang dilakukan untuk mengetahui kadar bilirubin total dalam tubuh seseorang. 
        - Direct Bilirubin  dilakukan untuk mengetahui kadar bilirubin direk di dalam darah.
        - Alkaline Phosphotase dilakukan untuk mengukur jumlah enzim alkaline phosphatase dalam aliran darah. 
        - Alanine Aminotransferase dapat mengukur jumlah enzim dalam darah. 
        - Aspartate Aminotransferase tes darah yang merupakan bagian dari organ hati.
        - Total Proteins dilakukan untuk mengukur jumlah total dua jenis protein pada tubuh yaitu albumin dan globulin
        - Albumin  dilakukan untuk melihat kadar albumin dalam darah.
        - Albumin and Globulin Ratio perbandingan albumin dengan globulin, yang merupakan konstituen utama protein yang ditemukan dalam darah.
        - Hasil: 1 Masalah Hati = Pasien penyakit hati (Positive), 2 Tidak Masalah Hati = Bukan pasien penyakit hati(Negative)
        """
    )

    st.subheader("""Sumber Dataset""")
    st.write("""
    Sumber data di dapatkan melalui website kaggle.com, Berikut merupakan link untuk mengakses sumber dataset.
    <a href="https://www.kaggle.com/datasets/gauravduttakiit/indian-liver-patient">Klik disini</a>""", unsafe_allow_html=True)
        
with dataset:
    st.subheader("""Dataset Anemia""")
    df = pd.read_csv('https://raw.githubusercontent.com/Alfito12/dataset/main/indian_liver_patient_dataset.csv')
    st.dataframe(df)

with preprocessing:
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    #Mendefinisikan Varible X dan Y
    
    Gender = df['Gender'].values
    ge = LabelEncoder()
    ge.fit(Gender)
    genderBaru = ge.transform(Gender)
    genderbaru = pd.DataFrame(genderBaru, columns=['gender'])
    new_df = pd.concat([df,genderbaru], axis = 1)
    new_df = new_df.dropna(axis=0)

    X = new_df.drop(columns=['Liver_Problem', 'Gender'])
    X
    y = new_df['Liver_Problem'].values
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.Liver_Problem).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        'Negative' : [dumies[1]],
        'Positive' : [dumies[0]]
    })

    st.write(labels)


with modeling:
    #Split Data 
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        #Gaussian Naive Bayes
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        probas = gaussian.predict_proba(test)
        probas = probas[:,1]
        probas = probas.round()

        gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier(random_state=1)
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)

        
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        Age = st.number_input('Umur')
        Gender = st.slider('Jenis Kelamin', 0,1)
        Total_Bilirubin = st.number_input('Total Bilirubin')
        Direct_Bilirubin = st.number_input('Direct Bilirubin')
        Alkaline_Phosphotase = st.number_input('Alkaline Phosphotase')
        Alamine_Aminotransferase = st.number_input('Alamine Aminotransferase')
        Aspartate_Aminotransferase = st.number_input('Aspartate Aminotransferase')
        Total_Protiens = st.number_input('Total Protiens')
        Albumin = st.number_input('Albumin')
        Albumin_and_Globulin_Ratio = st.number_input('Albumin dan Globulin Ratio')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                Age,
                Total_Bilirubin,
                Direct_Bilirubin,
                Alkaline_Phosphotase,
                Alamine_Aminotransferase,
                Aspartate_Aminotransferase,
                Total_Protiens,
                Albumin,
                Albumin_and_Globulin_Ratio,
                Gender,
            ])
            
            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt

            input_pred = mod.predict(input_norm)

            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            if input_pred == 1:
                st.error('Positive')
            else:
                st.success('Negative')
            
            

footer="""
<style>
a:link , 
a:visited{
color: white;
background-color: transparent;
}

a:hover,  
a:active {
color: Gainsboro;
background-color: transparent;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
height: 30px;
background-color: red;
color: white;
margin: 4;
text-align: center;
}
</style>

<div class="footer">
    <span>Copyright &copy; 2022 by <a href="mailto: Alfitow12@gmail.com">Alfito Wahyu Kamaly</a> All Right Reserved</span>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

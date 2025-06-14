import streamlit as st
st.set_page_config(page_title="Proyek Data Mining", layout="centered")
st.title("📊 Proyek Data Mining")

st.header("👥 Perkenalan Kelompok")
st.markdown("""Halo! 👋 Kami dari kelompok 1 Data Mining yang terdiri dari:
1. Dayinta Ajeng Nariswari (4101400048)
2. Firyal Daffa Nisrinna (4101422049)
3. Aulia Fitri Za'imah (4101422053)
4. Yarfak Rafiansya Saputra (4101422067)
5. Ani Rahayu (2304030003)

Kami sedang mengerjakan sebuah proyek yang menerapkan konsep-konsep data mining untuk analisis data nyata.
""")

st.header("🧑‍🏫 Dosen Pengampu")
st.markdown("""Mata Kuliah ini dibimbing oleh: Bapak M. Faris Al Hakim S.Pd., M.Cs.""")

st.header("🛠️ Fitur Aplikasi")
st.markdown("Aplikasi ini memiliki 3 fitur yaitu:")

st.page_link("pages/1_Dataset_Anemia.py", label="📈 Data Anemia", icon="📊")
st.page_link("pages/2_Prediksi.py", label="🧠 Prediksi Anemia", icon="🔮")
st.page_link("pages/3_Performance.py", label="📋 Performance Model", icon="📈")

st.markdown("---")
st.caption("Dibuat dengan ❤️ oleh Kelompok 1 Data Mining")

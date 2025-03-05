{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import pandas as pd\
import numpy as np\
import torch\
import emoji\
import os\
import tempfile\
import base64\
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline\
\
# Configuraci\'f3n de la p\'e1gina\
st.set_page_config(\
    page_title="An\'e1lisis de Sentimientos para Comentarios con Emojis",\
    page_icon="\uc0\u55357 \u56832 ",\
    layout="wide",\
)\
\
# T\'edtulo y descripci\'f3n\
st.title("An\'e1lisis de Sentimientos para Comentarios con Emojis")\
st.write("""\
Esta aplicaci\'f3n analiza los sentimientos de comentarios que pueden contener emojis,\
y los clasifica en positivos, negativos o neutros.\
""")\
\
@st.cache_resource\
def load_sentiment_model():\
    """\
    Cargar el modelo preentrenado de Hugging Face.\
    El cache_resource asegura que el modelo se cargue solo una vez.\
    """\
    try:\
        # Usar un modelo multiling\'fce preentrenado para an\'e1lisis de sentimientos\
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"\
        \
        # En producci\'f3n, es mejor descargar el modelo una vez y guardarlo localmente\
        sentiment_analyzer = pipeline(\
            "sentiment-analysis", \
            model=model_name, \
            tokenizer=model_name,\
            return_all_scores=True\
        )\
        \
        return sentiment_analyzer\
    except Exception as e:\
        st.error(f"Error al cargar el modelo: \{str(e)\}")\
        return None\
\
def preprocess_emoji_text(text):\
    """\
    Preprocesar texto para manejar emojis.\
    Convierte emojis a su descripci\'f3n textual y limpia el texto.\
    """\
    if not isinstance(text, str):\
        return ""\
    \
    # Convertir emojis a texto\
    text_with_emoji_desc = emoji.demojize(text)\
    \
    # Limpiar el formato de la descripci\'f3n del emoji\
    text_with_emoji_desc = text_with_emoji_desc.replace(':', ' ')\
    text_with_emoji_desc = text_with_emoji_desc.replace('_', ' ')\
    \
    return text_with_emoji_desc\
\
def map_score_to_sentiment(score):\
    """\
    Mapear la puntuaci\'f3n num\'e9rica del modelo a una etiqueta de sentimiento.\
    El modelo nlptown/bert-base-multilingual-uncased-sentiment usa una escala de 1-5 estrellas.\
    """\
    if score <= 2:\
        return "negative"\
    elif score >= 4:\
        return "positive"\
    else:\
        return "neutral"\
\
def analyze_sentiment(sentiment_analyzer, text):\
    """\
    Analizar el sentimiento de un texto usando el modelo preentrenado.\
    """\
    if not text or len(text.strip()) == 0:\
        return "neutral", 0.5, \{"negative": 0.33, "neutral": 0.34, "positive": 0.33\}\
    \
    # Preprocesar texto para manejar emojis\
    processed_text = preprocess_emoji_text(text)\
    \
    try:\
        # El modelo devuelve puntuaciones para 1-5 estrellas\
        results = sentiment_analyzer(processed_text)\
        \
        # Calcular puntuaci\'f3n ponderada (escala 1-5)\
        score = 0\
        for result in results[0]:\
            # La etiqueta es como "1 star", "2 stars", etc.\
            label = result['label']\
            stars = int(label.split()[0])\
            score += stars * result['score']\
        \
        # Normalizar a escala 0-1 para probabilidades\
        normalized_score = (score - 1) / 4.0\
        \
        # Mapear a sentimiento\
        sentiment = map_score_to_sentiment(score)\
        \
        # Calcular probabilidades aproximadas para cada clase\
        if sentiment == "negative":\
            probs = \{"negative": 0.6 + (1-normalized_score)*0.3, "neutral": 0.3 - (1-normalized_score)*0.2, "positive": 0.1\}\
        elif sentiment == "positive":\
            probs = \{"negative": 0.1, "neutral": 0.3 - normalized_score*0.2, "positive": 0.6 + normalized_score*0.3\}\
        else:\
            probs = \{"negative": 0.3, "neutral": 0.4, "positive": 0.3\}\
        \
        return sentiment, normalized_score, probs\
    except Exception as e:\
        st.error(f"Error al analizar sentimiento: \{str(e)\}")\
        return "neutral", 0.5, \{"negative": 0.33, "neutral": 0.34, "positive": 0.33\}\
\
def identify_comment_column(df):\
    """\
    Identifica autom\'e1ticamente la columna que contiene los comentarios.\
    """\
    possible_columns = ['comment', 'comments', 'texto', 'text', 'comentario', 'comentarios', 'message', 'mensaje']\
    \
    for col in possible_columns:\
        if col in df.columns:\
            return col\
    \
    # Si no encuentra ninguna de las columnas esperadas, usar la primera columna de texto\
    for col in df.columns:\
        if df[col].dtype == 'object':\
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""\
            if isinstance(sample, str) and len(sample) > 5:\
                return col\
    \
    # Si no encuentra ninguna columna adecuada\
    return None\
\
def process_file(sentiment_analyzer, uploaded_file):\
    """\
    Procesar un archivo subido y analizar los sentimientos.\
    """\
    # Determinar el tipo de archivo\
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()\
    \
    # Cargar el DataFrame seg\'fan el tipo de archivo\
    try:\
        if file_extension == '.csv':\
            df = pd.read_csv(uploaded_file)\
        elif file_extension in ['.xls', '.xlsx']:\
            df = pd.read_excel(uploaded_file)\
        else:\
            st.error(f"Formato de archivo no soportado: \{file_extension\}")\
            return None\
    except Exception as e:\
        st.error(f"Error al leer el archivo: \{str(e)\}")\
        return None\
    \
    # Identificar la columna de comentarios\
    comment_col = identify_comment_column(df)\
    if not comment_col:\
        st.error("No se pudo identificar una columna de comentarios en el archivo.")\
        return None\
    \
    st.info(f"Utilizando la columna '\{comment_col\}' para los comentarios.")\
    \
    # Asegurarse de que todos los valores sean strings\
    df[comment_col] = df[comment_col].astype(str)\
    \
    # Crear barras de progreso para mostrar el avance\
    progress_bar = st.progress(0)\
    status_text = st.empty()\
    \
    # Analizar los comentarios\
    sentiments = []\
    confidence_values = []\
    negative_probs = []\
    neutral_probs = []\
    positive_probs = []\
    \
    total_rows = len(df)\
    for i, comment in enumerate(df[comment_col]):\
        sentiment, confidence, probs = analyze_sentiment(sentiment_analyzer, comment)\
        sentiments.append(sentiment)\
        confidence_values.append(confidence)\
        negative_probs.append(probs["negative"])\
        neutral_probs.append(probs["neutral"])\
        positive_probs.append(probs["positive"])\
        \
        # Actualizar la barra de progreso\
        progress = (i + 1) / total_rows\
        progress_bar.progress(progress)\
        status_text.text(f"Analizando comentarios: \{i+1\}/\{total_rows\}")\
    \
    # A\'f1adir las columnas de sentimiento y confianza al DataFrame\
    df['sentiment'] = sentiments\
    df['confidence'] = confidence_values\
    df['prob_negative'] = negative_probs\
    df['prob_neutral'] = neutral_probs\
    df['prob_positive'] = positive_probs\
    \
    # Limpiar la UI\
    progress_bar.empty()\
    status_text.empty()\
    \
    return df\
\
def get_download_link(df, filename, text):\
    """\
    Generar un enlace de descarga para el DataFrame.\
    """\
    file_extension = os.path.splitext(filename)[1].lower()\
    \
    with tempfile.NamedTemporaryFile(delete=False) as tmp:\
        if file_extension == '.csv':\
            df.to_csv(tmp.name, index=False)\
        elif file_extension in ['.xlsx', '.xls']:\
            df.to_excel(tmp.name, index=False, engine='openpyxl')\
        else:\
            # Por defecto, usar CSV\
            df.to_csv(tmp.name, index=False)\
            filename = filename + '.csv'\
            \
    with open(tmp.name, 'rb') as f:\
        data = f.read()\
    \
    b64 = base64.b64encode(data).decode()\
    href = f'<a href="data:file/\{file_extension\};base64,\{b64\}" download="\{filename\}" target="_blank">\{text\}</a>'\
    return href\
\
def main():\
    # Cargar el modelo\
    with st.spinner("Cargando el modelo de an\'e1lisis de sentimientos..."):\
        sentiment_analyzer = load_sentiment_model()\
    \
    if sentiment_analyzer is None:\
        st.error("No se pudo cargar el modelo. Por favor, recarga la p\'e1gina.")\
        return\
    \
    # Secci\'f3n para cargar el archivo\
    uploaded_file = st.file_uploader("Sube tu archivo (CSV, XLS, XLSX)", type=["csv", "xls", "xlsx"])\
    \
    if uploaded_file is not None:\
        # Procesar el archivo\
        with st.spinner("Analizando sentimientos..."):\
            result_df = process_file(sentiment_analyzer, uploaded_file)\
        \
        if result_df is not None:\
            # Mostrar vista previa de los resultados\
            st.subheader("Vista previa de los resultados")\
            st.dataframe(result_df.head(10))\
            \
            # Estad\'edsticas de los sentimientos\
            st.subheader("Distribuci\'f3n de sentimientos")\
            sentiment_counts = result_df['sentiment'].value_counts()\
            sentiment_percentages = sentiment_counts / len(result_df) * 100\
            \
            col1, col2 = st.columns(2)\
            with col1:\
                st.write("Conteo por sentimiento:")\
                for sentiment, count in sentiment_counts.items():\
                    st.write(f"- \{sentiment\}: \{count\}")\
            \
            with col2:\
                st.write("Porcentaje por sentimiento:")\
                for sentiment, percentage in sentiment_percentages.items():\
                    st.write(f"- \{sentiment\}: \{percentage:.2f\}%")\
            \
            # Proporcionar un enlace para descargar el archivo con los resultados\
            output_filename = "sentimiento_" + uploaded_file.name\
            download_link = get_download_link(result_df, output_filename, "Descargar resultados")\
            st.markdown(download_link, unsafe_allow_html=True)\
            \
            # Secci\'f3n para probar comentarios individuales\
            st.subheader("Probar comentarios individuales")\
            test_comment = st.text_area("Ingresa un comentario para analizar:")\
            if test_comment:\
                sentiment, confidence, probs = analyze_sentiment(sentiment_analyzer, test_comment)\
                st.write(f"Sentimiento: **\{sentiment\}** (confianza: \{confidence:.4f\})")\
                st.write("Probabilidades:")\
                for label, prob in probs.items():\
                    st.write(f"- \{label\}: \{prob:.4f\}")\
    \
    # Instrucciones adicionales\
    else:\
        st.info("""\
        ### Instrucciones:\
        1. Sube un archivo CSV o Excel que contenga una columna con comentarios.\
        2. La aplicaci\'f3n detectar\'e1 autom\'e1ticamente la columna de comentarios.\
        3. Espera mientras se analiza el sentimiento de cada comentario.\
        4. Descarga el archivo con los resultados.\
        \
        ### Tipos de sentimientos:\
        - **positive**: Comentarios con tono positivo o favorable.\
        - **negative**: Comentarios con tono negativo o cr\'edtico.\
        - **neutral**: Comentarios neutros o informativos sin carga emocional clara.\
        \
        ### Ejemplo de comentarios:\
        - "\'a1Me encanta este producto! \uc0\u55357 \u56845 " \u8594  positivo\
        - "No funciona como esperaba \uc0\u55357 \u56864 " \u8594  negativo\
        - "Lleg\'f3 el paquete ayer" \uc0\u8594  neutral\
        """)\
\
if __name__ == "__main__":\
    main()}
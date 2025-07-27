import streamlit as st
import requests
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import smtplib
from email.message import EmailMessage
import sqlite3
import pandas as pd
from datetime import datetime, date
import os
from dotenv import load_dotenv
import plotly.express as px
import io
import base64

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
DEFAULT_EMAIL_TO = os.getenv("EMAIL_TO", "authority@example.com")

# Validate environment variables
if not all([API_KEY, EMAIL_FROM, EMAIL_PASSWORD]):
    st.error("Missing critical environment variables. Please check your .env file.")
    st.stop()

# OpenWeatherMap API setup
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# Initialize SQLite database (fallback for PostgreSQL)
def init_db():
    """Initialize SQLite database for fire risk logs and FWI data."""
    try:
        conn = sqlite3.connect("fire_logs.db")
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS fire_logs
                     (timestamp TEXT, city TEXT, temp REAL, humidity REAL, ffmc REAL, dmc REAL, dc REAL, isi REAL, risk TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS fwi_data
                     (city TEXT, date TEXT, ffmc REAL, dmc REAL, dc REAL, isi REAL)''')
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error initializing database: {str(e)}")
        st.stop()

init_db()

# Cache model loading
@st.cache_resource
def load_model():
    """Load the forest fire risk prediction model."""
    try:
        model = joblib.load("forest_fire_risk_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()
if model is None:
    st.error("Cannot proceed without a valid model.")
    st.stop()

def kelvin_to_celsius(kelvin):
    """Convert temperature from Kelvin to Celsius."""
    return kelvin - 273.15

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_weather_data(city_name):
    """
    Fetch weather data from OpenWeatherMap API.
    
    Args:
        city_name (str): Name of the city.
    
    Returns:
        tuple: (weather dict, error message)
    """
    params = {"q": city_name, "appid": API_KEY}
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if response.status_code != 200:
            return None, data.get("message", "Unknown error")

        temp = kelvin_to_celsius(data.get('main', {}).get('temp', 0))
        humidity = data.get('main', {}).get('humidity', 0)
        wind_speed = data.get('wind', {}).get('speed', 0)
        rainfall = data.get('rain', {}).get('1h', 0)

        return {
            "temp": round(temp, 2) if temp else 0,
            "humidity": humidity,
            "wind": wind_speed,
            "rain": rainfall
        }, None
    except requests.RequestException as e:
        return None, f"Error fetching weather data: {str(e)}"

def calculate_fwi(temp, humidity, wind, rain):
    """
    Mock FWI calculation (placeholder for pyfwi or real API).
    
    Args:
        temp (float): Temperature in Celsius.
        humidity (float): Relative humidity percentage.
        wind (float): Wind speed in m/s.
        rain (float): Rainfall in mm.
    
    Returns:
        dict: Estimated FWI parameters.
    """
    # TODO: Replace with pyfwi or real FWI API (e.g., Copernicus CDS)
    # Simplified mock calculation based on typical FWI ranges
    ffmc = min(90 + (temp * 0.5 - humidity * 0.2), 100)  # Fine Fuel Moisture Code
    dmc = min(15 + (temp * 0.3 - rain * 2), 50)          # Duff Moisture Code
    dc = min(25 + (temp * 0.4 - rain * 3), 100)          # Drought Code
    isi = min(wind * 0.8 + ffmc * 0.1, 20)               # Initial Spread Index
    return {'FFMC': ffmc, 'DMC': dmc, 'DC': dc, 'ISI': isi}

def fetch_fwi_data(city, date_str):
    """
    Fetch FWI data (FFMC, DMC, DC, ISI) for a city and date.
    
    Args:
        city (str): Name of the city.
        date_str (str): Date in YYYY-MM-DD format.
    
    Returns:
        dict: FWI parameters or calculated values as fallback.
    """
    # Option 1: Check SQLite database
    try:
        conn = sqlite3.connect("fire_logs.db")
        c = conn.cursor()
        c.execute("SELECT ffmc, dmc, dc, isi FROM fwi_data WHERE city = ? AND date = ?",
                  (city, date_str))
        result = c.fetchone()
        conn.close()
        if result:
            return {'FFMC': result[0], 'DMC': result[1], 'DC': result[2], 'ISI': result[3]}
    except Exception as e:
        st.warning(f"Error fetching FWI from database: {str(e)}")

    # Option 2: Fetch weather data and calculate FWI (mock)
    weather, error = get_weather_data(city)
    if weather:
        return calculate_fwi(weather['temp'], weather['humidity'], weather['wind'], weather['rain'])
    
    # Fallback: Default values
    st.warning(f"No FWI data for {city}. Using default values.")
    return {'FFMC': 90, 'DMC': 15, 'DC': 25, 'ISI': 10}

def predict_fire_risk(temp, humidity, ffmc, dmc, dc, isi, month, day):
    """
    Predict forest fire risk based on weather and FWI parameters.
    
    Args:
        temp (float): Temperature in Celsius.
        humidity (float): Relative humidity percentage.
        ffmc (float): Fine Fuel Moisture Code.
        dmc (float): Duff Moisture Code.
        dc (float): Drought Code.
        isi (float): Initial Spread Index.
        month (int): Month of the year (1-12).
        day (int): Day of the week (0-6, Monday=0).
    
    Returns:
        str: Predicted fire risk level ('Low', 'Medium', 'High').
    """
    sample_input = np.array([[temp, humidity, ffmc, dmc, dc, isi, month, day]])
    try:
        prediction = model.predict(sample_input)[0]
        valid_risks = ["Low", "Medium", "High"]
        return prediction if prediction in valid_risks else "Unknown"
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Unknown"

@st.cache_data(ttl=86400)  # Cache for 1 day
def get_coordinates(city):
    """
    Get latitude and longitude for a city using Nominatim.
    
    Args:
        city (str): Name of the city.
    
    Returns:
        tuple: (latitude, longitude) or (None, None) if not found.
    """
    geolocator = Nominatim(user_agent="forest_fire_predictor_app")
    try:
        location = geolocator.geocode(city, timeout=10)
        return (location.latitude, location.longitude) if location else (None, None)
    except Exception as e:
        st.error(f"Error fetching coordinates: {str(e)}")
        return None, None

def send_email_alert(city, fire_risk, recipients, threshold="High"):
    """
    Send email alert based on fire risk level.
    
    Args:
        city (str): Name of the city.
        fire_risk (str): Predicted fire risk level.
        recipients (list): List of recipient email addresses.
        threshold (str): Risk level to trigger alert (default: High).
    
    Returns:
        tuple: (success bool, error message)
    """
    if fire_risk != threshold:
        return True, None
    msg = EmailMessage()
    msg['Subject'] = f"🔥 ALERT: {fire_risk} Forest Fire Risk in {city}"
    msg['From'] = EMAIL_FROM
    msg['To'] = ", ".join(recipients)
    msg.set_content(f"""
    {fire_risk} fire risk detected in {city} on {date.today()}.
    Immediate attention required from local authorities.
    """)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_FROM, EMAIL_PASSWORD)
            smtp.send_message(msg)
        return True, None
    except Exception as e:
        return False, f"Email failed to send: {str(e)}"

def log_fire_risk(city, weather, fwi_data, fire_risk):
    """
    Log fire risk data to SQLite database.
    
    Args:
        city (str): Name of the city.
        weather (dict): Weather data.
        fwi_data (dict): FWI parameters.
        fire_risk (str): Predicted fire risk level.
    """
    try:
        conn = sqlite3.connect("fire_logs.db")
        c = conn.cursor()
        c.execute("INSERT INTO fire_logs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (datetime.now().isoformat(), city, weather['temp'], weather['humidity'],
                   fwi_data['FFMC'], fwi_data['DMC'], fwi_data['DC'], fwi_data['ISI'], fire_risk))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error logging to database: {str(e)}")

def validate_city(city):
    """
    Validate city input.
    
    Args:
        city (str): Name of the city.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    return bool(city and isinstance(city, str) and len(city.strip()) >= 2)

# Multi-language support
LANGUAGES = {
    "en": {
        "title": "🔥 Forest Fire Prediction System",
        "description": "Predict forest fire risk based on real-time weather and FWI data.",
        "city_input": "Enter City Name",
        "city_placeholder": "e.g., London",
        "invalid_city": "Please enter a valid city name (at least 2 characters).",
        "fetching_weather": "Fetching live weather data...",
        "weather_success": "Weather data fetched successfully!",
        "weather_info": "📊 Live Weather Info",
        "temp_label": "🌡 Temperature: *{temp}°C*",
        "humidity_label": "💧 Humidity: *{humidity}%*",
        "wind_label": "💨 Wind Speed: *{wind} m/s*",
        "rain_label": "🌧 Rainfall: *{rain} mm*",
        "fwi_label": "🔥 FWI Parameters: FFMC={ffmc}, DMC={dmc}, DC={dc}, ISI={isi}",
        "risk_level": "🔥 Fire Risk Level",
        "low_risk": "✅ Low Risk – Safe",
        "medium_risk": "⚠ Medium Risk – Be Aware",
        "high_risk": "🚨 High Risk – Alert Authorities",
        "unknown_risk": "🚨 Unknown Risk Level – Model Error",
        "email_success": "🚨 Email alert sent to local authorities.",
        "map_title": "📍 Fire Risk Map",
        "map_error": "Could not generate map for this location.",
        "logs_title": "📁 View Past Risk Logs",
        "no_logs": "No logs available yet.",
        "logs_error": "Error reading logs: {error}",
        "trends_title": "📈 View Risk Trends",
        "no_trends": "No trend data available.",
        "export_logs": "Download Logs as CSV",
        "alert_config": "Configure Email Alerts",
        "recipients_label": "Email Recipients (comma-separated)",
        "threshold_label": "Alert Threshold",
        "about": """
        ### About
        This app predicts forest fire risk using the Canadian Forest Fire Weather Index (FWI) and real-time weather data.
        Enter a city to get risk levels, view maps, and receive alerts for high-risk conditions.
        """
    },
    "es": {
        "title": "🔥 Sistema de Predicción de Incendios Forestales",
        "description": "Predice el riesgo de incendios forestales basado en datos meteorológicos y FWI en tiempo real.",
        "city_input": "Ingrese el Nombre de la Ciudad",
        "city_placeholder": "ej., Madrid",
        "invalid_city": "Por favor, ingrese un nombre de ciudad válido (mínimo 2 caracteres).",
        "fetching_weather": "Obteniendo datos meteorológicos en tiempo real...",
        "weather_success": "¡Datos meteorológicos obtenidos con éxito!",
        "weather_info": "📊 Información Meteorológica en Tiempo Real",
        "temp_label": "🌡 Temperatura: *{temp}°C*",
        "humidity_label": "💧 Humedad: *{humidity}%*",
        "wind_label": "💨 Velocidad del Viento: *{wind} m/s*",
        "rain_label": "🌧 Precipitación: *{rain} mm*",
        "fwi_label": "🔥 Parámetros FWI: FFMC={ffmc}, DMC={dmc}, DC={dc}, ISI={isi}",
        "risk_level": "🔥 Nivel de Riesgo de Incendio",
        "low_risk": "✅ Riesgo Bajo – Seguro",
        "medium_risk": "⚠ Riesgo Medio – Esté Atento",
        "high_risk": "🚨 Riesgo Alto – Alerte a las Autoridades",
        "unknown_risk": "🚨 Nivel de Riesgo Desconocido – Error del Modelo",
        "email_success": "🚨 Alerta por correo enviada a las autoridades.",
        "map_title": "📍 Mapa de Riesgo de Incendio",
        "map_error": "No se pudo generar el mapa para esta ubicación.",
        "logs_title": "📁 Ver Registros Anteriores",
        "no_logs": "No hay registros disponibles aún.",
        "logs_error": "Error al leer registros: {error}",
        "trends_title": "📈 Ver Tendencias de Riesgo",
        "no_trends": "No hay datos de tendencias disponibles.",
        "export_logs": "Descargar Registros como CSV",
        "alert_config": "Configurar Alertas por Correo",
        "recipients_label": "Destinatarios de Correo (separados por comas)",
        "threshold_label": "Umbral de Alerta",
        "about": """
        ### Acerca de
        Esta aplicación predice el riesgo de incendios forestales utilizando el Índice Meteorológico de Incendios Forestales de Canadá (FWI) y datos meteorológicos en tiempo real.
        Ingrese una ciudad para obtener niveles de riesgo, ver mapas y recibir alertas para condiciones de alto riesgo.
        """
    }
}

# Streamlit App UI
st.set_page_config(page_title="Forest Fire Predictor", layout="wide", initial_sidebar_state="expanded")

# Sidebar for language selection and settings
with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Language / Idioma", ["English", "Spanish"], index=0)
    lang_code = "en" if language == "English" else "es"
    texts = LANGUAGES[lang_code]
    
    st.subheader(texts["alert_config"])
    email_recipients = st.text_input(texts["recipients_label"], value=DEFAULT_EMAIL_TO)
    alert_threshold = st.selectbox(texts["threshold_label"], ["High", "Medium", "High"], index=0)
    
    st.markdown(texts["about"])

st.title(texts["title"])
st.markdown(texts["description"])

# Predefined cities for multi-city support
CITIES = ["London", "Toronto", "Sydney", "Vancouver", "Calgary", "Madrid", "Athens"]
city = st.selectbox(texts["city_input"], options=[""] + CITIES, format_func=lambda x: texts["city_placeholder"] if x == "" else x)

if city:
    if not validate_city(city):
        st.error(texts["invalid_city"])
    else:
        with st.spinner(texts["fetching_weather"]):
            weather, error = get_weather_data(city)

        if weather:
            st.success(texts["weather_success"])

            # Display weather info
            st.subheader(texts["weather_info"])
            st.write(texts["temp_label"].format(temp=weather['temp']))
            st.write(texts["humidity_label"].format(humidity=weather['humidity']))
            st.write(texts["wind_label"].format(wind=weather['wind']))
            st.write(texts["rain_label"].format(rain=weather['rain']))

            # Fetch FWI data
            with st.spinner("Fetching FWI data..."):
                fwi_data = fetch_fwi_data(city, date.today().isoformat())
                st.write(texts["fwi_label"].format(
                    ffmc=fwi_data['FFMC'], dmc=fwi_data['DMC'], dc=fwi_data['DC'], isi=fwi_data['ISI']
                ))

            # Predict fire risk
            fire_risk = predict_fire_risk(
                weather['temp'], weather['humidity'], fwi_data['FFMC'], fwi_data['DMC'],
                fwi_data['DC'], fwi_data['ISI'], datetime.now().month, datetime.now().weekday()
            )

            # Display fire risk
            st.subheader(texts["risk_level"])
            if fire_risk == "Low":
                st.success(texts["low_risk"])
            elif fire_risk == "Medium":
                st.warning(texts["medium_risk"])
            elif fire_risk == "High":
                st.error(texts["high_risk"])
            else:
                st.error(texts["unknown_risk"])

            # Send email alert
            recipients = [r.strip() for r in email_recipients.split(",") if r.strip()]
            if recipients:
                success, email_error = send_email_alert(city, fire_risk, recipients, alert_threshold)
                if success:
                    st.info(texts["email_success"])
                else:
                    st.error(email_error)

            # Log the prediction
            log_fire_risk(city, weather, fwi_data, fire_risk)

            # Display map
            st.subheader(texts["map_title"])
            with st.spinner("Generating map..."):
                lat, lon = get_coordinates(city)
                if lat and lon:
                    m = folium.Map(location=[lat, lon], zoom_start=8)
                    color = {"Low": "green", "Medium": "orange", "High": "red", "Unknown": "blue"}.get(fire_risk, "blue")
                    folium.Marker(
                        location=[lat, lon],
                        popup=f"{city}: {fire_risk} Risk",
                        icon=folium.Icon(color=color),
                    ).add_to(m)
                    st_folium(m, width=700, height=450)
                else:
                    st.warning(texts["map_error"])

        else:
            st.error(f"Could not fetch weather data: {error}")

# Display past logs
if st.checkbox(texts["logs_title"]):
    try:
        conn = sqlite3.connect("fire_logs.db")
        logs = pd.read_sql_query("SELECT * FROM fire_logs", conn)
        conn.close()
        if logs.empty:
            st.info(texts["no_logs"])
        else:
            logs.columns = ["Timestamp", "City", "Temperature (°C)", "Humidity (%)", "FFMC", "DMC", "DC", "ISI", "Risk Level"]
            st.dataframe(logs.style.format({"Temperature (°C)": "{:.2f}", "FFMC": "{:.1f}", "DMC": "{:.1f}", "DC": "{:.1f}", "ISI": "{:.1f}"}))
            
            # Export logs as CSV
            csv = logs.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="fire_logs.csv">{texts["export_logs"]}</a>'
            st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(texts["logs_error"].format(error=str(e)))

# Display risk trends
if st.checkbox(texts["trends_title"]):
    try:
        conn = sqlite3.connect("fire_logs.db")
        logs = pd.read_sql_query("SELECT * FROM fire_logs WHERE city = ?", conn, params=(city,))
        conn.close()
        if logs.empty or not city:
            st.info(texts["no_trends"])
        else:
            logs['Timestamp'] = pd.to_datetime(logs['Timestamp'])
            fig = px.line(logs, x='Timestamp', y='Risk Level', title=f"Fire Risk Trends for {city}",
                          labels={"Risk Level": "Risk Level", "Timestamp": "Date"})
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating trends: {str(e)}")
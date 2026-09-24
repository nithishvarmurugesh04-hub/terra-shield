from fastapi import FastAPI
from routes import reports, gis
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone
import joblib
import pandas as pd
import httpx

app = FastAPI(
    title="TERRA-SHIELD API",
    description="AI-powered landslide risk monitoring system",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LATITUDE = 25.6751
LONGITUDE = 94.1086

alerts_store: list[dict] = []
reports_store: list[dict] = []
# Load the trained model once at startup.
ml_bundle = joblib.load("ml/risk_model.pkl")
ml_model = ml_bundle["model"]
ml_feature_columns = ml_bundle["feature_columns"]

# Static terrain/history values for the demo location. A production
# version would pull slope/elevation from a DEM (digital elevation model)
# API and historical_landslide_count from a records database — out of
# scope for this prototype, so we use representative fixed values here
# while the ML model handles fusing them with live weather data.
DEMO_SLOPE = 32.0
DEMO_ELEVATION = 1444.0
DEMO_HISTORICAL_COUNT = 4

class ReportIn(BaseModel):
    """
    Defines the exact shape of data a report submission must have.
    FastAPI uses this to automatically validate incoming POST requests —
    if the frontend sends a malformed request (e.g. missing 'description'),
    FastAPI rejects it with a clear error before your code even runs.
    """
    report_type: str
    description: str
    latitude: float
    longitude: float


async def fetch_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": "temperature_2m,precipitation",
        "hourly": "precipitation,soil_moisture_0_to_1cm",
        "timezone": "auto",
    }
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        current = data["current"]
        hourly = data["hourly"]

        rainfall_24h = round(sum(hourly["precipitation"][-24:]), 1)
        soil_moisture = round(hourly["soil_moisture_0_to_1cm"][-1] * 100, 1)
        temperature = current.get("temperature_2m", 0)

        return {
            "rainfall_24h": rainfall_24h,
            "soil_moisture": soil_moisture,
            "temperature": temperature,
            "source": "open-meteo",
        }

    except Exception:
        return {
            "rainfall_24h": 0,
            "soil_moisture": 0,
            "temperature": 0,
            "source": "fallback-demo-data",
        }


def calculate_risk(weather: dict):
    """
    Builds a feature row from live weather + static terrain/history
    values, and asks the trained LightGBM model for a risk probability.
    """
    features = pd.DataFrame([{
        "rainfall_24h": weather["rainfall_24h"],
        "rainfall_72h": weather["rainfall_24h"] * 1.6,  # rough proxy; see note below
        "soil_moisture": weather["soil_moisture"],
        "slope": DEMO_SLOPE,
        "elevation": DEMO_ELEVATION,
        "historical_landslide_count": DEMO_HISTORICAL_COUNT,
    }])[ml_feature_columns]

    # predict_proba returns [P(no landslide), P(landslide)] — we want the second.
    risk_score = round(float(ml_model.predict_proba(features)[0][1]), 2)

    if risk_score < 0.3:
        level = "LOW"
    elif risk_score < 0.6:
        level = "MODERATE"
    elif risk_score < 0.8:
        level = "HIGH"
    else:
        level = "CRITICAL"

    return risk_score, level


def maybe_create_alert(location: str, risk_score: float, risk_level: str):
    if risk_level not in ("HIGH", "CRITICAL"):
        return

    already_alerted = any(
        a["risk_level"] == risk_level and a["location"] == location
        for a in alerts_store
    )
    if already_alerted:
        return

    alerts_store.append({
        "id": len(alerts_store) + 1,
        "location": location,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "message": f"{risk_level} landslide risk detected near {location}.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.get("/")
def root():
    return {"system": "TERRA-SHIELD", "status": "online"}


@app.get("/api/v1/health")
def health():
    return {"status": "healthy", "service": "TERRA-SHIELD backend"}


@app.get("/api/v1/weather")
async def get_weather():
    return await fetch_weather()


@app.get("/api/v1/risk")
async def get_risk(force_level: str | None = None):
    weather = await fetch_weather()
    risk_score, risk_level = calculate_risk(weather)
    location = "Kohima, Nagaland (demo zone)"

    if force_level in ("LOW", "MODERATE", "HIGH", "CRITICAL"):
        risk_level = force_level
        risk_score = {"LOW": 0.15, "MODERATE": 0.45, "HIGH": 0.7, "CRITICAL": 0.9}[force_level]

    maybe_create_alert(location, risk_score, risk_level)

    return {
        "location": location,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "confidence": 0.7,
        "model": "lightgbm-v1-synthetic",
    }


@app.get("/api/v1/alerts")
def get_alerts():
    return {"alerts": list(reversed(alerts_store))}


@app.get("/api/v1/reports")
def get_reports():
    return {"reports": list(reversed(reports_store))}


@app.post("/api/v1/reports")
def create_report(report: ReportIn):
    """
    FastAPI parses the incoming JSON body into a validated ReportIn
    object automatically, just from the type hint 'report: ReportIn'.
    No manual JSON parsing or validation code needed.
    """
    new_report = {
        "id": len(reports_store) + 1,
        "report_type": report.report_type,
        "description": report.description,
        "latitude": report.latitude,
        "longitude": report.longitude,
        "status": "Pending verification",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    reports_store.append(new_report)
    return new_report

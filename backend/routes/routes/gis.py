from fastapi import APIRouter

router = APIRouter()


@router.get("/gis/risk-zones")
def get_risk_zones():
    return {
        "type": "FeatureCollection",
        "features": []
    }


@router.get("/gis/roads")
def get_roads():
    return {
        "type": "FeatureCollection",
        "features": []
    }


@router.get("/gis/villages")
def get_villages():
    return {
        "type": "FeatureCollection",
        "features": []
    }

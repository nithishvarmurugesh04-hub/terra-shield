from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class FieldReport(BaseModel):
    latitude: float
    longitude: float
    report_type: str
    description: str
    photo_url: str | None = None


reports = []


@router.get("/reports")
def get_reports():
    return {"reports": reports}


@router.post("/reports")
def create_report(report: FieldReport):
    reports.append(report.model_dump())
    return {
        "message": "Report received",
        "report": report
    }

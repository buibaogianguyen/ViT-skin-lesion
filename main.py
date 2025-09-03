from fastapi import FastAPI
from data.data_manager import SkinLesion, init_db
from sqlalchemy import func
from fastapi.responses import HTMLResponse

app = FastAPI()

def get_db():
    db_session = init_db()
    try:
        yield db_session
    finally:
        db_session.close()

@app.get("/stats")

def stats():
    db = next(get_db())
    total = db.query(SkinLesion).count()
    per_label = dict(db.query(SkinLesion.label, func.count()).group_by(SkinLesion.label).all())
    avg_age = db.query(func.avg(SkinLesion.age)).scalar()

    return {"total": total, "per_label": per_label, "avg_age": avg_age}
        

@app.get('/lesion/{image_id}')

def get_lesion(image_id):
    db_session = next(get_db())
    lesion = db_session.query(SkinLesion).filter_by(image_id=image_id).first()
    if not lesion:
        return {"error": "Not found"}
    
    return {
        "image_id": lesion.image_id,
        "label": lesion.label,
        "age": lesion.age,
        "sex": lesion.sex,
        "anatom_site_general": lesion.anatom_site_general,
        "mel_prob": lesion.mel_prob,
        "nv_prob": lesion.nv_prob,
        "bcc_prob": lesion.bcc_prob,
        "akiec_prob": lesion.akiec_prob,
        "bkl_prob": lesion.bkl_prob,
        "df_prob": lesion.df_prob,
        "vasc_prob": lesion.vasc_prob,
        "scc_prob": lesion.scc_prob,
        "unk_prob": lesion.unk_prob,
    }
from data.data_manager import SkinLesion, init_db, store_preds
import os

def store_to_db(image_id, probs):
    db_session = init_db()

    lesion = db_session.query(SkinLesion).filter_by(image_id=image_id).first()

    if not lesion:
        lesion = SkinLesion(
            image_id=image_id,
            label="unk",
            age=None,
            sex=None,
            anatom_site_general=None
        )
        db_session.add(lesion)
        db_session.commit()

    store_preds(image_id, probs, db_session)

    print(f"Stored predictions for image {image_id} in skin_lesions.db")

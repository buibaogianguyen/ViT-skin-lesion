from data.data_manager import SkinLesion, load_metadata, init_db, store_preds
import os

def store_to_db(probs):
    db_session = init_db()

    image_folder = load_metadata(db_session)

    for filename in os.listdir(image_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_id = os.path.splitext(filename)[0]
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

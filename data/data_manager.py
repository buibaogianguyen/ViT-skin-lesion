import sqlalchemy
from sqlalchemy import create_engine, Column, String, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from data.data_downloader import download_dataset
import os
import pandas as pd

Base = declarative_base()

class SkinLesion(Base):
    __tablename__ = 'skin_lesions'
    image_id = Column(String, primary_key=True)
    label = Column(String)
    age = Column(Integer, nullable=True)
    sex = Column(String, nullable=True)
    localization = Column(String, nullable=True)
    mel_prob = Column(Float, nullable=True)
    bcc_prob = Column(Float, nullable=True)
    akiec_prob = Column(Float, nullable=True)
    bkl_prob = Column(Float, nullable=True)
    df_prob = Column(Float, nullable=True)
    vasc_prob = Column(Float, nullable=True)
    nv_prob = Column(Float, nullable=True)

def init_db(db_path='sqlite:///skin_lesions.db'):
    engine = create_engine(db_path, echo=False)
    Base.metadata.create_all(engine)

    return sessionmaker(bind=engine)()

def load_metadata(db_session):
    dataset_path = download_dataset()
    metadata_path = os.path.join(dataset_path, 'HAM10000_metadata.csv')

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Run download_dataset.py or recheck for missing files/file corruptions.")
    
    df = pd.read_csv(metadata_path)
    
    if 'image_id' not in df.columns:
        raise ValueError("Metadata CSV file is missing required column: image_id")
    
    if 'dx' not in df.columns:
        raise ValueError("Metadata CSV file is missing required column: dx")
    
    for _, row in df.iterrows():
        lesion = SkinLesion(
            image_id = row['image_id'],
            label = row['dx'],
            age = int(row['age']) if pd.notna(row['age']) else None,
            sex = row['sex'] if pd.notna(row['sex']) else None,
            localization = row['localization'] if pd.notna(row['localization']) else None
        )
        db_session.merge(lesion)

    db_session.commit()
    
    return dataset_path

def store_preds(image_id, probs, db_session):
    lesion = db_session.query(SkinLesion).filter_by(image_id=image_id).first()

    if not lesion:
        raise ValueError(f"Image ID {image_id} not found in database.")
    
    lesion.mel_prob = float(probs[0])
    lesion.bcc_prob = float(probs[1])
    lesion.akiec_prob = float(probs[2])
    lesion.bkl_prob = float(probs[3])
    lesion.df_prob = float(probs[4])
    lesion.vasc_prob = float(probs[5])
    lesion.nv_prob = float(probs[6])
    db_session.commit()
    
    print(f"Stored predictions for image id: {image_id}")

if __name__ == '__main__':
    db_session = init_db()

    try:
        dataset_path = load_metadata(db_session)
    except Exception as e:
        print(f'Error loading metadata: {e}')

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
    anatom_site_general = Column(String, nullable=True)
    mel_prob = Column(Float, nullable=True)
    nv_prob = Column(Float, nullable=True)
    bcc_prob = Column(Float, nullable=True)
    akiec_prob = Column(Float, nullable=True)
    bkl_prob = Column(Float, nullable=True)
    df_prob = Column(Float, nullable=True)
    vasc_prob = Column(Float, nullable=True)
    scc_prob = Column(Float, nullable=True)
    unk_prob = Column(Float, nullable=True)

def init_db(db_path='sqlite:///skin_lesions.db'):
    engine = create_engine(db_path, echo=False)
    Base.metadata.create_all(engine)

    return sessionmaker(bind=engine)()

def load_metadata(db_session):
    dataset_path = download_dataset()
    metadata_path = os.path.join(dataset_path, 'ISIC_2019_Training_Metadata.csv')
    groundtruth_path = os.path.join(dataset_path, 'ISIC_2019_Training_GroundTruth.csv')

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Run download_dataset.py or recheck for missing files/file corruptions.")
    if not os.path.exists(groundtruth_path):
        raise FileNotFoundError(f"Ground truth file not found at {groundtruth_path}. Run download_dataset.py or recheck for missing files/file corruptions.")
    
    metadata_df = pd.read_csv(metadata_path)
    groundtruth_df = pd.read_csv(groundtruth_path)
    
    if 'image_id' not in df.columns:
        raise ValueError("Metadata CSV file is missing required column: image_id")

    df = metadata_df.merge(groundtruth_df, left_on='image', right_on='image', how='inner') 
    
    label_columns = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    df['label'] = df[label_columns].idxmax(axis=1).map({
        'MEL': 'mel', 'NV': 'nv', 'BCC': 'bcc', 'AK': 'akiec',
        'BKL': 'bkl', 'DF': 'df', 'VASC': 'vasc', 'SCC': 'scc', 'UNK': 'unk'
    })
    
    for _, row in df.iterrows():
        lesion = SkinLesion(
            image_id = row['image'],
            label = row['label'],
            age = int(row['age_approx']) if pd.notna(row['age_approx']) else None,
            sex = row['sex'] if pd.notna(row['sex']) else None,
            anatom_site_general = row['anatom_site_general'] if pd.notna(row['anatom_site_general']) else None
        )
        db_session.merge(lesion)

    db_session.commit()
    print(f"Loaded {len(df)} records into database.")

    return dataset_path

def store_preds(image_id, probs, db_session):
    lesion = db_session.query(SkinLesion).filter_by(image_id=image_id).first()

    if not lesion:
        raise ValueError(f"Image ID {image_id} not found in database.")
    
    lesion.mel_prob = float(probs[0])
    lesion.nv_prob = float(probs[1])
    lesion.bcc_prob = float(probs[2])
    lesion.akiec_prob = float(probs[3])
    lesion.bkl_prob = float(probs[4])
    lesion.df_prob = float(probs[5])
    lesion.vasc_prob = float(probs[6])
    lesion.scc_prob = float(probs[7])
    lesion.unk_prob = float(probs[8])
    db_session.commit()
    
    print(f"Stored predictions for image id: {image_id}")

if __name__ == '__main__':
    db_session = init_db()

    try:
        dataset_path = load_metadata(db_session)
    except Exception as e:
        print(f'Error loading metadata: {e}')

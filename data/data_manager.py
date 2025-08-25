import sqlalchemy
from sqlalchemy import create_engine, Column, String, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

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

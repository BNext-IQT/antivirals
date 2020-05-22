import enum
from typing import Sequence, Dict
import pandas as pd
from sqlalchemy import Column, Integer, String, ForeignKey, Enum, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.session import Session
from sqlalchemy.engine.base import Connectable

Base = declarative_base()


class OriginCategory(enum.Enum):
    GroundTruth = 1
    Computed = 2
    Interactive = 3
    Unknown = 4


class PartitionCategory(enum.Enum):
    Train = 1
    Test = 2
    Verify = 3
    Unspecific = 4


class Test(enum.Enum):
    PassFail = 1
    StringIdentifier = 2
    IntIdentifier = 3
    Intensity = 4
    Threshold = 5
    NotApplicable = 10


class PropertyCategory(enum.Enum):
    Toxicity = 1
    Manufacturablity = 2
    Solubility = 3
    Novelty = 4
    Bioactivity = 5
    DataSetIdentifier = 6
    Tag = 7


class Property(Base):
    __tablename__ = 'properties'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    desc = Column(String, nullable=False)
    test = Column(Enum(Test), nullable=False)
    category = Column(Enum(PropertyCategory), nullable=False)
    threshold = Column(Float)


class PassFailMetadata(Base):
    __tablename__ = 'metadata_passfail'
    id = Column(Integer, primary_key=True)
    prop_id = Column(Integer, ForeignKey('properties.id'))
    mol_id = Column(Integer, ForeignKey('molecules.id'))
    value = Column(Boolean)
    confidence = Column(Float)


class TextMetadata(Base):
    __tablename__ = 'metadata_text'
    id = Column(Integer, primary_key=True)
    prop_id = Column(Integer, ForeignKey('properties.id'))
    mol_id = Column(Integer, ForeignKey('molecules.id'))
    value = Column(String, nullable=False)


class Origin(Base):
    __tablename__ = 'origins'
    id = Column(Integer, primary_key=True)
    category = Column(Enum(OriginCategory), nullable=False)
    name = Column(String, nullable=False)
    desc = Column(String, nullable=False)


class Molecule(Base):
    __tablename__ = 'molecules'
    id = Column(Integer, primary_key=True)
    origin_id = Column(Integer, ForeignKey('origins.id'))
    partition = Column(Enum(PartitionCategory), nullable=False)
    smiles = Column(String, nullable=False)


class Molecules:
    def __init__(self, sess: Session):
        self.sess = sess
        self._create_prop_dict()

    def add(self, orgin_id: int, smiles: str, metadata: Dict = None,
            partition: PartitionCategory = PartitionCategory.Unspecific):
        """
        Add a molecular descriptor and its metadata to the database.
        """
        mol = Molecule(origin_id=orgin_id, smiles=smiles, partition=partition)
        self.sess.add(mol)
        self.sess.flush()
        if metadata:
            self._map_metadata(mol.id, metadata)

    def commit(self):
        self.sess.commit()

    def get_all_mols(self) -> pd.Series:
        """
        Get all registered molecular descriptors.
        """
        query = 'select molecules.smiles from molecules'
        return pd.read_sql(query, self.sess.bind)['smiles']

    def get_mols_with_passfail_labels(self) -> pd.DataFrame:
        """
        Get all PassFail (binary) classification problems mapped to their molecular descriptors.
        """
        query = 'select molecules.smiles, properties.name, metadata_passfail.value from molecules inner join metadata_passfail, properties ON metadata_passfail.mol_id=molecules.id and metadata_passfail.prop_id=properties.id'
        df = pd.read_sql(query, self.sess.bind)
        return df.pivot_table(
            index=['smiles'],
            columns=['name'],
            values='value')

    def _map_metadata(self, mol_id: int, metadata: Dict):
        """
        Maps a metadata dictionary into the chemical relational model.
        """
        for k, v in metadata.items():
            if isinstance(v, float):
                prop_id = self.props.get(k)
                if prop_id:
                    self.sess.add(PassFailMetadata(
                        prop_id=prop_id, mol_id=mol_id, value=bool(v)))
            elif isinstance(v, str):
                prop_id = self.props.get(k)
                if prop_id:
                    self.sess.add(TextMetadata(
                        prop_id=prop_id, mol_id=mol_id, value=v))
            elif isinstance(v, list):
                prop_id = self.props.get(k)
                if prop_id:
                    for i in v:
                        self.sess.add(TextMetadata(
                            prop_id=prop_id, mol_id=mol_id,
                            value=i))
            else:
                prop_id = self.props.get(k)
                if prop_id:
                    value, confidence = v
                    self.sess.add(PassFailMetadata(
                        prop_id=prop_id, mol_id=mol_id, value=bool(value),
                        confidence=confidence))

    def _create_prop_dict(self):
        """
        Create a lookup table for the metadata mapper.
        """
        self.props = {}
        props_query = self.sess.query(Property).all()

        for prop in props_query:
            self.props[prop.name] = prop.id


def add_props(sess: Session, props: Sequence):
    """
    Add properties to the database that don't already exist.
    """
    for prop in props:
        existing = sess.query(Origin).filter(
            Property.name == prop.name).first()
        if not existing:
            sess.add(prop)
    sess.flush()


class DatasetExistsError(Exception):
    pass


def add_dataset(sess: Session, dataset: Origin) -> int:
    """
    Add a dataset to the database if it doesn't already exist.
    """
    existing = sess.query(Origin).filter(Origin.name == dataset.name).first()
    if existing:
        raise DatasetExistsError(
            f"{existing.name} already exists in the database with index {existing.id}")
    sess.add(dataset)
    sess.flush()
    return dataset.id


def create_db(engine: Connectable):
    """
    Create the database schema.
    """
    Base.metadata.create_all(engine)

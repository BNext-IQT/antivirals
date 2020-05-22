import pickle
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm.session import sessionmaker, Session
from antivirals.schema import Molecules
from antivirals.chem import Chemistry
from antivirals.data import download_all


def _create_sess(dbstring: str) -> Session:
    engine = create_engine(dbstring)
    return sessionmaker()(bind=engine)


def run_agent(dbstring: str):
    pass


def run_data_gathering(dbstring: str):
    download_all(_create_sess(dbstring))


def run_train_models(dbstring: str):
    sess = _create_sess(dbstring)
    mols = Molecules(sess)
    chem = Chemistry()
    chem.from_molecules(mols)

    Path('data').mkdir(exist_ok=True)
    with open(Path('data', 'chemistry.model'), 'wb') as fd:
        pickle.dump(chem, fd)

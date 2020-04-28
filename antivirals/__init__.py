from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm.session import sessionmaker, Session
from antivirals.schema import Molecule, Molecules
from antivirals.chem import Language, Toxicity
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

    lm = Language()
    tox_data = mols.get_mols_with_passfail_labels()
    X = tox_data.index
    y = tox_data.astype('int')
    lm.fit(mols.get_all_mols(), X, y)

    Path('data/models').mkdir(parents=True, exist_ok=True)

    lm.save('data/models/language.model')

    tox = Toxicity(lm)
    tox.fit(X, y)

    tox.save('data/models/toxicity.model')


def run_do_science():
    pass

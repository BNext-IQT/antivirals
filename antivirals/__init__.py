import pickle
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm.session import sessionmaker, Session
from antivirals.schema import Molecules
from antivirals.chem import Chemistry, Hyperparameters
from antivirals.data import download_all

def _create_sess(dbstring: str) -> Session:
    engine = create_engine(dbstring)
    return sessionmaker()(bind=engine)


def _introspect_model(chem: Chemistry):
    print(f"Model UUID: {chem.uuid} :: "
          f"AUC: {sum(chem.toxicity.auc) / len(chem.toxicity.auc)} :: "
          f"Hyperparams: {chem.hyperparams.__dict__}")


def run_agent(dbstring: str):
    pass


def run_data_gathering(dbstring: str):
    download_all(_create_sess(dbstring))


def run_train_models(dbstring: str, hp = Hyperparameters()) -> Chemistry:
    sess = _create_sess(dbstring)
    mols = Molecules(sess)
    chem = Chemistry(hp)
    chem.from_molecules(mols)

    Path('data', 'chemistry').mkdir(parents=True, exist_ok=True)
    with open(Path('data', 'chemistry', chem.uuid), 'wb') as fd:
        pickle.dump(chem, fd)

    return chem


def garbage_collect_models(save_best_n: int, dry_run: bool, verbose: bool):
    model_dir = Path('data', 'chemistry')
    metrics = {}
    if verbose:
        print("MODEL INFORMATION\n"
              "-----------------\n")
    for model in model_dir.iterdir():
        with open(model, 'rb') as fd:
            try:
                chem = pickle.load(fd)
            except (pickle.UnpicklingError, EOFError):
                print(f"WARNING: Unpickling {model} failed. Skipping...")
                continue
            
            metrics[chem.uuid] = sum(chem.toxicity.auc) / len(chem.toxicity.auc)
            if verbose:
                _introspect_model(chem)
    
    models_sorted = [k for k, _ in sorted(metrics.items(), key=lambda i: i[1])]

    if verbose:
        print("\nGARBAGE COLLECTED MODELS\n"
              "------------------------")
    for model in models_sorted[:-save_best_n]:
        print(model)
        if not dry_run:
            (model_dir / model).unlink()
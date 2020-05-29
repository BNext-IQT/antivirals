import os
import sys
from fire import Fire
from antivirals import run_agent, run_data_gathering, run_train_models
from antivirals.chem import Hyperparameters


def _get_dbstring() -> str:
    db = os.environ.get("ANTIVIRALS_DB")

    return db


class Controller:
    def up(self, db=None):
        """
        Sets up the whole system and runs the agent. The kitchen sink command.
        """
        if not db:
            db = "sqlite://"
        run_data_gathering(db)
        run_train_models(db)
        run_agent(db)

    def gather(self, db=None):
        """
        Just download the datasets and create the molecular database.
        """
        if not db:
            db = "sqlite://"
        run_data_gathering(db)

    def train(self, db=None, hp=None):
        """
        Just train the cheminformatics models.
        """

        if not db:
            db = "sqlite://"
        if hp:
            run_train_models(db, Hyperparameters.from_dict(hp))
        else:
            run_train_models(db)

    def agent(self, db):
        """
        Just run an agent that searches for coronavirus antivirals.
        """
        run_agent(db)

    def agent_from_env(self):
        """
        Run an agent using the database string in the ANTIVIRALS_DB environment variable.
        """
        db = os.environ.get("ANTIVIRALS_DB")
        if not db:
            raise EnvironmentError("ANTIVIRALS_DB needs to be set.")

        run_agent(_get_dbstring())


def main():
    Fire(Controller)

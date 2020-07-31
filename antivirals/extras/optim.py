from sigopt import Connection
from sigopt.exception import ApiException
from antivirals import run_train_models
from antivirals.chem import Hyperparameters


def create_experiment_lda(apikey):
    conn = Connection(client_token=apikey)
    experiment = conn.experiments().create(
        name="Coronavirus Antivirals -- LDA",
        parameters=[
            dict(
                name="topics",
                bounds=dict(
                    min=16,
                    max=512
                ),
                type="int"
            ),
            dict(
                name="estimators",
                bounds=dict(
                    min=256,
                    max=512
                ),
                type="int"
            ),
            dict(
                name="topic_epochs",
                bounds=dict(
                    min=1,
                    max=6
                ),
                type="int"
            ),
            dict(
                name="decay",
                bounds=dict(
                    min=0.5,
                    max=0.999
                ),
                type="double"
            ),
            dict(
                name="max_vocab",
                bounds=dict(
                    min=10000,
                    max=100000
                ),
                type="int"
            ),
            dict(
                name="min_samples_split",
                bounds=dict(
                    min=4,
                    max=18
                ),
                type="int"
            ),
            dict(
                name="min_samples_leaf",
                bounds=dict(
                    min=2,
                    max=4
                ),
                type="int"
            ),
            dict(
                name="topic_iterations",
                bounds=dict(
                    min=20,
                    max=200
                ),
                type="int"
            ),
            dict(
                name="tree_criterion",
                categorical_values=[
                    dict(
                        name="gini"
                    ),
                    dict(
                        name="entropy"
                    )
                ],
                type="categorical"
            )
        ],
        metadata=dict(
            template="antivirals"
        ),
        observation_budget=300,
        parallel_bandwidth=10,
        project="antivirals"
    )
    return experiment.id


def create_experiment_doc2vec(apikey):
    conn = Connection(client_token=apikey)
    experiment = conn.experiments().create(
        name="Coronavirus Antivirals -- Doc2Vec",
        parameters=[
            dict(
                name="vec_dims",
                bounds=dict(
                    min=256,
                    max=512
                ),
                type="int"
            ),
            dict(
                name="estimators",
                bounds=dict(
                    min=256,
                    max=512
                ),
                type="int"
            ),
            dict(
                name="doc_epochs",
                bounds=dict(
                    min=70,
                    max=120
                ),
                type="int"
            ),
            dict(
                name="max_vocab",
                bounds=dict(
                    min=10000,
                    max=100000
                ),
                type="int"
            ),
            dict(
                name="vec_window",
                bounds=dict(
                    min=2,
                    max=4
                ),
                type="int"
            ),
            dict(
                name="min_samples_split",
                bounds=dict(
                    min=4,
                    max=18
                ),
                type="int"
            ),
            dict(
                name="min_samples_leaf",
                bounds=dict(
                    min=2,
                    max=4
                ),
                type="int"
            ),
            dict(
                name="tree_criterion",
                categorical_values=[
                    dict(
                        name="gini"
                    ),
                    dict(
                        name="entropy"
                    )
                ],
                type="categorical"
            )
        ],
        metadata=dict(
            template="antivirals"
        ),
        observation_budget=200,
        parallel_bandwidth=12,
        project="antivirals"
    )
    return experiment.id


def clean_experiment(apikey, exp_id):
    conn = Connection(client_token=apikey)
    conn.experiments(exp_id).suggestions().delete()


def continue_experiment(dbstring, apikey, exp_id):
    conn = Connection(client_token=apikey)
    experiment = conn.experiments(exp_id).fetch()

    for _ in range(experiment.observation_budget):
        try:
            suggestion = conn.experiments(exp_id).suggestions().create()
        except ApiException:
            conn.experiments(exp_id).suggestions().delete()
            suggestion = conn.experiments(exp_id).suggestions().create()
        if suggestion.assignments.get('topics'):
            suggestion.assignments['vector_algo'] = 'lda'
        elif suggestion.assignments.get('vec_dims'):
            suggestion.assignments['vector_algo'] = 'doc2vec'
        assignments = Hyperparameters.from_dict(suggestion.assignments)
        chem = run_train_models(dbstring, assignments)
        mean = sum(chem.toxicity.auc) / len(chem.toxicity.auc)
        conn.experiments(exp_id).observations().create(
            suggestion=suggestion.id,
            value=mean
        )

from sqlalchemy.orm.session import Session
import pandas as pd
from tqdm import tqdm
from antivirals.schema import DatasetExistsError, Molecules, Origin, OriginCategory, PropertyCategory, PartitionCategory, Property, Test, add_dataset, create_db, add_props

NR_AR = Property(
    name='NR-AR',
    desc='qHTS assay to identify small molecule agonists of the androgen receptor (AR) signaling pathway',
    category=PropertyCategory.Toxicity,
    test=Test.PassFail
)

NR_AR_LBD = Property(
    name='NR-AR-LBD',
    desc='qHTS assay to identify small molecule agonists of the androgen receptor (AR) signaling pathway using the MDA cell line',
    category=PropertyCategory.Toxicity,
    test=Test.PassFail
)

NR_AhR = Property(
    name='NR-AhR',
    desc='qHTS assay to identify small molecule that activate the aryl hydrocarbon receptor (AhR) signaling pathway',
    category=PropertyCategory.Toxicity, 
    test=Test.PassFail
)

NR_Aromatase = Property(
    name='NR-Aromatase',
    desc='qHTS assay to identify aromatase inhibitors',
    category=PropertyCategory.Toxicity,
    test=Test.PassFail
)

NR_ER = Property(
    name='NR-ER',
    desc='qHTS assay to identify small molecule agonists of the estrogen receptor alpha (ER-alpha) signaling pathway',
    category=PropertyCategory.Toxicity,
    test=Test.PassFail
)


NR_ER_LBD = Property(
    name='NR-ER-LBD',
    desc='qHTS assay to identify small molecule agonists of the estrogen receptor alpha (ER-alpha) signaling pathway using the BG1 cell line',
    category=PropertyCategory.Toxicity,
    test=Test.PassFail
)

NR_PPAR_gamma = Property(
    name='NR-PPAR-gamma',
    desc='qHTS assay to identify small molecule agonists of the peroxisome proliferator-activated receptor gamma (PPARg) signaling pathway',
    category=PropertyCategory.Toxicity,
    test=Test.PassFail
)


SR_ARE = Property(
    name='SR-ARE',
    desc='qHTS assay for small molecule agonists of the antioxidant response element (ARE) signaling pathway',
    category=PropertyCategory.Toxicity,
    test=Test.PassFail
)


SR_ATAD5 = Property(
    name='SR-ATAD5',
    desc='qHTS assay for small molecules that induce genotoxicity in human embryonic kidney cells expressing luciferase-tagged ATAD5',
    category=PropertyCategory.Toxicity,
    test=Test.PassFail
)


SR_HSE = Property(
    name='SR-HSE',
    desc='qHTS assay for small molecule activators of the heat shock response signaling pathway',
    category=PropertyCategory.Toxicity,
    test=Test.PassFail
)

SR_MMP = Property(
    name='SR-MMP',
    desc='qHTS assay for small molecule disruptors of the mitochondrial membrane potential',
    category=PropertyCategory.Toxicity,
    test=Test.PassFail
)


SR_p53 = Property(
    name='SR-p53',
    desc='qHTS assay for small molecule agonists of the p53 signaling pathway',
    category=PropertyCategory.Toxicity,
    test=Test.PassFail
)


ToxId = Property(
    name='ToxId',
    desc='The molecule ID in the Tox21 challenge dataset',
    category=PropertyCategory.DataSetIdentifier,
    test=Test.IntIdentifier
)

ChemblId = Property(
    name='ChEMBL_Id',
    desc='The molecule ID in the ChEMBL dataset',
    category=PropertyCategory.DataSetIdentifier,
    test=Test.IntIdentifier
)

ZincId = Property(
    name='ZINC_Id',
    desc='The molecule ID in the ZINC dataset',
    category=PropertyCategory.DataSetIdentifier,
    test=Test.IntIdentifier
)

Tag = Property(
    name='Tag',
    desc='Human-readable tag that is a superset of tags from the ZINC dataset',
    category=PropertyCategory.Tag,
    test=Test.StringIdentifier
)


class ZINC:
    origin = Origin(
        name='ZINC',
        desc='Collection of commerically available chemicals prepared for virtual screening',
        category=OriginCategory.GroundTruth
    )

    def to_sql(self, sess: Session):
        pass


class ChEMBL:
    origin = Origin(
        name='ChEMBL',
        desc='ChEMBL is a manually curated chemical database of bioactive molecules with drug-like properties.',
        category=OriginCategory.GroundTruth
    )
    source = 'ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_27_sqlite.tar.gz'
    archive_path = 'chembl_27/chembl_27_sqlite/chembl_27.db'

    def to_sql(self, sess: Session):
        pass

class MOSES:
    origin = Origin(
        name='MOSES',
        desc='Benchmark dataset of drug-like molecules from the ZINC Clean Leads collection',
        category=OriginCategory.GroundTruth
    )
    properties = []
    source = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv'

    def to_df(self) -> pd.DataFrame:
        return pd.read_csv(self.source)

    def to_sql(self, sess: Session):
        source_id = add_dataset(sess, self.origin)
        mols = Molecules(sess)
        df = self.to_df()
        row_count, _ = df.shape
        for _, row in tqdm(df.iterrows(), total=row_count, unit=' row'):
            if row.SPLIT == 'train':
                partition = PartitionCategory.Train
            elif row.SPLIT == 'test':
                partition = PartitionCategory.Test
            else:
                partition = PartitionCategory.Unspecific
            mols.add(source_id, row.SMILES, {'Tag': 'MOSES'}, partition)

        mols.commit()


class Tox21:
    origin = Origin(
        name='Tox21',
        desc='Qualitative toxicity measurements including nuclear receptors and stress response pathways',
        category=OriginCategory.GroundTruth
    )
    properties = [NR_AhR, NR_AR, NR_AR_LBD, NR_Aromatase, NR_ER, NR_ER_LBD,
                  NR_PPAR_gamma, SR_ARE, SR_ATAD5, SR_HSE, SR_MMP, SR_p53]
    source = 'https://github.com/deepchem/deepchem/raw/master/datasets/tox21.csv.gz'

    def to_df(self) -> pd.DataFrame:
        return pd.read_csv(self.source)

    def to_sql(self, sess: Session):
        source_id = add_dataset(sess, self.origin)
        add_props(sess, self.properties)
        mols = Molecules(sess)
        df = self.to_df()
        row_count, _ = df.shape
        for _, row in tqdm(df.iterrows(), total=row_count, unit=' row'):
            mols.add(source_id, row.smiles, row.to_dict())
        mols.commit()


def download_all(sess: Session):
    create_db(sess.bind)
    datasets = [MOSES(), Tox21()]
    for dataset in datasets:
        try:
            print(f"{dataset.origin.name} - {dataset.origin.desc}")
            dataset.to_sql(sess)
        except DatasetExistsError:
            print(f"{dataset.origin.name} already exists in the database... skipping.")

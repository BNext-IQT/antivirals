'''
A system test intended to stress the antivirals up codepath, while being much faster and easier to detect bugs with.
'''

from sqlalchemy.orm.session import Session, sessionmaker
from sqlalchemy import create_engine
from antivirals.schema import add_dataset, Molecules, OriginCategory, Origin, PartitionCategory, Molecule, TextMetadata, PassFailMetadata, create_db, add_props
from antivirals.data import Tag, NR_AR
from antivirals.chem import Language


class FakeDataset:
    origin = Origin(
        name='Fake',
        desc='Just for testing.',
        category=OriginCategory.Dataset
    )
    props = [Tag, NR_AR]

    def to_sql(self, sess: Session):
        source_id = add_dataset(sess, self.origin)
        add_props(sess, self.props)
        mols = Molecules(sess)
        mols.add(source_id, 'CN1CCC[C@H]1c2cccnc2', {
                 'Tag': 'Test'}, PartitionCategory.Unspecific)
        mols.add(source_id, 'O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5', {
                 'Tag': ['Test1', 'Test2'], 'NR-AR': 1.0}, PartitionCategory.Verify)
        mols.commit()


SESS = sessionmaker()(bind=create_engine('sqlite://'))


def test_dataset():
    create_db(SESS.bind)
    FakeDataset().to_sql(SESS)
    assert SESS.query(Molecule).count() == 2
    assert SESS.query(TextMetadata).count() == 3
    assert SESS.query(PassFailMetadata).count() == 1

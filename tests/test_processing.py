import unittest
from rdkit import Chem

from src.data import processing, constants


class TestProcessing(unittest.TestCase):
    def test_one_hot_encode(self):
        self.assertEqual(
            processing.one_hot_encode(
                0, constants.AtomFeatures.ALLOWED_VALUES
            ).tolist(),
            [1] + [0] * (len(constants.AtomFeatures.ALLOWED_VALUES) - 1),
        )
        self.assertEqual(
            processing.one_hot_encode(
                13, constants.AtomFeatures.ALLOWED_VALUES
            ).tolist(),
            [0] * len(constants.AtomFeatures.ALLOWED_VALUES),
        )

    def test_one_hot_encode_with_unknown(self):
        self.assertEqual(
            processing.one_hot_encode_with_unknown(
                constants.AtomFeatures.CHARATOMSET[0],
                constants.AtomFeatures.CHARATOMSET,
            ).tolist(),
            [1] + [0] * (len(constants.AtomFeatures.CHARATOMSET) - 1),
        )
        self.assertEqual(
            processing.one_hot_encode_with_unknown(
                "X", constants.AtomFeatures.CHARATOMSET
            ).tolist(),
            [0] * (len(constants.AtomFeatures.CHARATOMSET) - 1) + [1],
        )

    def test_get_atom_features(self):
        atom = Chem.MolFromSmiles("CC").GetAtomWithIdx(0)
        self.assertEqual(
            len(processing.get_atom_features(atom)),
            len(constants.AtomFeatures.CHARATOMSET)
            + len(constants.AtomFeatures.ALLOWED_VALUES) * 3
            + 1,
        )

    def test_smile_to_graph(self):
        c_size, features, edge_index = processing.smile_to_graph("CC")
        self.assertEqual(c_size, 2)
        self.assertEqual(len(features), 2)
        self.assertEqual(len(edge_index), 2)

    def test_tokenize_sequence(self):
        encoding = processing.tokenize_sequence("ABC", constants.Tokens.CHARPROTSET, 5)
        self.assertEqual(encoding.tolist(), [1.0, 3.0, 2.0, 0.0, 0.0])

    def test_tokenize_smiles(self):
        encoding = processing.tokenize_smiles("CC", 5)
        self.assertEqual(len(encoding), 5)

        encoding = processing.tokenize_smiles("XX", 5)
        self.assertEqual(encoding.tolist(), [0.0] * 5)

        encoding = processing.tokenize_smiles("CZi", 5, isomeric=True)
        self.assertEqual(encoding.tolist(), [42.0, 19.0, 59.0, 0.0, 0.0])

    def test_tokenize_target(self):
        encoding = processing.tokenize_target("ABC", 5)
        self.assertEqual(len(encoding), 5)

        encoding = processing.tokenize_target("JJJ", 5)
        self.assertEqual(len(encoding), 5)


if __name__ == "__main__":
    unittest.main()

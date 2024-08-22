import unittest
from rdkit import Chem
import numpy as np

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

    def test_tokenise_sequence(self):
        encoding = processing.tokenise_sequence("ABC", constants.Tokens.CHARPROTSET, 5)
        self.assertEqual(encoding.tolist(), [1.0, 3.0, 2.0, 0.0, 0.0])

    def test_tokenise_smiles(self):
        encoding = processing.tokenise_smiles("CC", 5)
        self.assertEqual(len(encoding), 5)

        encoding = processing.tokenise_smiles("XX", 5)
        self.assertEqual(encoding.tolist(), [0.0] * 5)

        encoding = processing.tokenise_smiles("CZi", 5, to_isomeric=False)
        self.assertEqual(encoding.tolist(), [42.0, 19.0, 59.0, 0.0, 0.0])

    def test_tokenise_target(self):
        encoding = processing.tokenise_target("ABC", 5)
        self.assertEqual(len(encoding), 5)

        encoding = processing.tokenise_target("JJJ", 5)
        self.assertEqual(len(encoding), 5)


class TestToDeepSmiles(unittest.TestCase):
    def test_simple_smiles_conversion(self):
        smiles = "CCO"
        expected_deep_smiles = "CCO"
        self.assertEqual(processing.to_deepsmiles(smiles), expected_deep_smiles)

    def test_complex_smiles_conversion(self):
        smiles = "C1=CC=CC=C1"
        expected_deep_smiles = "C=CC=CC=C6"
        self.assertEqual(processing.to_deepsmiles(smiles), expected_deep_smiles)

    def test_no_rings_branches_smiles_conversion(self):
        smiles = "CCCC"
        expected_deep_smiles = "CCCC"
        self.assertEqual(processing.to_deepsmiles(smiles), expected_deep_smiles)


class TestSeqToWords(unittest.TestCase):
    def test_short_sequence(self):
        sequence = "MVKVYAPAS"
        word_len = 3
        expected = ("MVK", "VYA", "PAS", "VKV", "YAP", "KVY", "APA")
        result = processing.seq_to_words(sequence, word_len, max_length=10)
        self.assertEqual(result, expected)

    def test_longer_sequence(self):
        sequence = "MVKVYAPASSANMSVGFDVLGAAVTPVD"
        word_len = 4
        expected = (
            "MVKV",
            "YAPA",
            "SSAN",
        )
        result = processing.seq_to_words(sequence, word_len, max_length=3)
        self.assertEqual(result, expected)


class TestMakeWordsDict(unittest.TestCase):
    def test_with_single_sequence(self):
        sequences = [("MVK", "VYA", "PAS")]
        expected_dict = {"MVK": 1, "VYA": 2, "PAS": 3}
        result = processing.make_words_dict(sequences)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result, dict)

    def test_with_multiple_sequences(self):
        sequences = [("MVK", "VYA"), ("PAS", "VKV")]
        expected_keys = {"MVK", "VYA", "PAS", "VKV"}
        result = processing.make_words_dict(sequences)
        self.assertEqual(len(result), 4)
        self.assertTrue(all(key in result for key in expected_keys))

    def test_with_empty_sequence(self):
        sequences = []
        result = processing.make_words_dict(sequences)
        self.assertEqual(len(result), 0)

    def test_with_repeated_words(self):
        sequences = [("MVK", "VYA"), ("MVK", "VYA"), ("VYA", "MVK")]
        expected_dict_length = 2
        result = processing.make_words_dict(sequences)
        self.assertEqual(len(result), expected_dict_length)

    def test_with_different_length_sequences(self):
        sequences = [("MVK",), ("VYA", "PAS"), ("VKV", "YAP", "KVY")]
        expected_dict_length = 6
        result = processing.make_words_dict(sequences)
        self.assertEqual(len(result), expected_dict_length)


class TestEncodeWords(unittest.TestCase):
    def setUp(self):
        self.word_to_int = {"A": 1, "B": 2, "C": 3}
        self.length = 5

    def test_encode_word_all_known(self):
        sequence = ["A", "B", "C"]
        expected = [1, 2, 3, 0, 0]
        result = processing.encode_word(sequence, self.word_to_int, self.length)
        self.assertEqual(result.tolist(), expected)

    def test_encode_word_with_unknown(self):
        sequence = ["A", "X", "C"]
        expected = [1, 0, 3, 0, 0]
        result = processing.encode_word(sequence, self.word_to_int, self.length)
        self.assertEqual(result.tolist(), expected)

    def test_encode_word_empty_sequence(self):
        sequence = []
        expected = [0, 0, 0, 0, 0]
        result = processing.encode_word(sequence, self.word_to_int, self.length)
        self.assertEqual(result.tolist(), expected)

    def test_encode_word_longer_than_length(self):
        sequence = ["A", "B", "C", "A", "B", "C"]
        expected = [1, 2, 3, 1, 2]
        result = processing.encode_word(sequence, self.word_to_int, self.length)
        self.assertEqual(result.tolist(), expected)

    def test_encode_word_shorter_than_length(self):
        sequence = ["A", "B"]
        expected = [1, 2, 0, 0, 0]
        result = processing.encode_word(sequence, self.word_to_int, self.length)
        self.assertEqual(result.tolist(), expected)


if __name__ == "__main__":
    unittest.main()

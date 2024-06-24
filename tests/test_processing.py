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

    def test_tokenize_sequence(self):
        encoding = processing.tokenize_sequence("ABC", constants.Tokens.CHARPROTSET, 5)
        self.assertEqual(encoding.tolist(), [1.0, 3.0, 2.0, 0.0, 0.0])

    def test_tokenize_smiles(self):
        encoding = processing.tokenize_smiles("CC", 5)
        self.assertEqual(len(encoding), 5)

        encoding = processing.tokenize_smiles("XX", 5)
        self.assertEqual(encoding.tolist(), [0.0] * 5)

        encoding = processing.tokenize_smiles("CZi", 5, to_isomeric=False)
        self.assertEqual(encoding.tolist(), [42.0, 19.0, 59.0, 0.0, 0.0])

    def test_tokenize_target(self):
        encoding = processing.tokenize_target("ABC", 5)
        self.assertEqual(len(encoding), 5)

        encoding = processing.tokenize_target("JJJ", 5)
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
        expected = {
            "MVK",
            "VKV",
            "KVY",
            "VYA",
            "YAP",
            "APA",
            "PAS",
        }
        result = set(processing.seq_to_words(sequence, word_len))
        self.assertEqual(result, expected)

    def test_longer_sequence(self):
        sequence = "MVKVYAPASSANMSVGFDVLGAAVTPVD"
        word_len = 4
        expected = {
            "MVKV",
            "VKVY",
            "KVYA",
            "VYAP",
            "YAPA",
            "APAS",
            "PASS",
            "ASSA",
            "SSAN",
            "SANM",
            "ANMS",
            "NMSV",
            "MSVG",
            "SVGF",
            "VGFD",
            "GFDV",
            "FDVL",
            "DVLG",
            "VLGA",
            "LGAA",
            "GAAV",
            "AAVT",
            "AVTP",
            "VTPV",
            "TPVD",
        }
        result = set(processing.seq_to_words(sequence, word_len))
        self.assertEqual(result, expected)


class TestMakeWordsSet(unittest.TestCase):
    def test_empty_input(self):
        sequences = []
        expected = set()
        self.assertEqual(processing.make_words_set(sequences), expected)

    def test_single_sequence(self):
        sequences = [("ABC", "DEF")]
        expected = {"ABC", "DEF"}
        self.assertEqual(processing.make_words_set(sequences), expected)

    def test_multiple_sequences(self):
        sequences = [("ABC", "DEF"), ("GHI", "JKL")]
        expected = {"ABC", "DEF", "GHI", "JKL"}
        self.assertEqual(processing.make_words_set(sequences), expected)

    def test_duplicate_words_across_sequences(self):
        sequences = [("ABC", "DEF"), ("DEF", "GHI")]
        expected = {"ABC", "DEF", "GHI"}
        self.assertEqual(processing.make_words_set(sequences), expected)


class TestOneHotWords(unittest.TestCase):
    def setUp(self):
        self.allowable_set = ["A", "C", "G", "T"]  # Example for DNA sequences
        self.length = 5

    def test_simple_input(self):
        input_sequence = ["A", "C", "G", "A", "T"]
        expected_output = [1, 2, 3, 1, 4]
        self.assertEqual(
            processing.one_hot_words(
                input_sequence, self.allowable_set, self.length
            ).tolist(),
            expected_output,
        )

    def test_input_with_unknown_words(self):
        input_sequence = ["A", "X", "G", "Y", "T"]
        expected_output = [1, 0, 3, 0, 4]
        self.assertEqual(
            processing.one_hot_words(
                input_sequence, self.allowable_set, self.length
            ).tolist(),
            expected_output,
        )

    def test_empty_input(self):
        input_sequence = []
        expected_output = [0, 0, 0, 0, 0]
        self.assertEqual(
            processing.one_hot_words(
                input_sequence, self.allowable_set, self.length
            ).tolist(),
            expected_output,
        )

    def test_output_length_matches_specified_length(self):
        input_sequence = ["A", "C"]
        output = processing.one_hot_words(
            input_sequence, self.allowable_set, self.length
        )
        self.assertEqual(len(output), self.length)


if __name__ == "__main__":
    unittest.main()

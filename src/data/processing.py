"""
Functions for labeling and encoding chemical characters like Compound SMILES and atom string, refer to
https://github.com/hkmztrk/DeepDTA and https://github.com/thinng/GraphDTA.
"""

import logging

import numpy as np
from rdkit import Chem
import networkx as nx
import deepsmiles

from .constants import Tokens, AtomFeatures


# Functions --------------------------------------------------------------------
def one_hot_encode(x, allowable_set) -> np.array:
    if x not in allowable_set:
        logging.warning(f"Input {x} not in allowable set {allowable_set}.")
        return np.zeros(len(allowable_set), dtype=int)

    return np.array([x == s for s in allowable_set], dtype=int)


def one_hot_encode_with_unknown(x, allowable_set) -> np.array:
    x = x if x in allowable_set else allowable_set[-1]

    return np.array([x == s for s in allowable_set], dtype=int)


def get_atom_features(atom) -> np.array:
    symbol_encoding = one_hot_encode_with_unknown(
        atom.GetSymbol(), AtomFeatures.CHARATOMSET
    )

    degree_encoding = one_hot_encode(atom.GetDegree(), AtomFeatures.ALLOWED_VALUES)

    num_h_encoding = one_hot_encode_with_unknown(
        atom.GetTotalNumHs(), AtomFeatures.ALLOWED_VALUES
    )

    valence_encoding = one_hot_encode_with_unknown(
        atom.GetImplicitValence(), AtomFeatures.ALLOWED_VALUES
    )

    aromatic_encoding = np.array([atom.GetIsAromatic()], dtype=int)

    return np.concatenate(
        [
            symbol_encoding,
            degree_encoding,
            num_h_encoding,
            valence_encoding,
            aromatic_encoding,
        ]
    )


def smile_to_graph(smiles: str):
    """ """
    mol = Chem.MolFromSmiles(smiles)
    c_size = mol.GetNumAtoms()

    features = np.array([get_atom_features(atom) for atom in mol.GetAtoms()])
    features = features / features.sum(axis=1, keepdims=True)

    edges = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]

    di_graph = nx.Graph(edges).to_directed()
    edge_index = list(di_graph.edges)

    return c_size, features.tolist(), edge_index


def tokenize_sequence(sequence: str, char_set: dict, max_length: int = 85) -> np.array:
    """Tokenizes a sequence using a given character set."""
    sequence_array = np.array(list(sequence[:max_length]))
    encoding = np.zeros(max_length)
    encoding[: len(sequence_array)] = np.vectorize(char_set.get)(sequence_array, 0)

    return encoding


def tokenize_smiles(
    smiles: str, max_length: int = 85, to_isomeric: bool = False
) -> np.array:
    """Tokenizes a SMILES string."""
    if to_isomeric:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.warning(f"rdkit cannot find this SMILES {smiles}.")
            return np.zeros(max_length)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    return tokenize_sequence(smiles, Tokens.CHARISOSMISET, max_length)


def tokenize_target(sequence: str, max_length: int = 1200) -> np.array:
    """Tokenizes a protein sequence."""

    return tokenize_sequence(sequence.upper(), Tokens.CHARPROTSET, max_length)


# WideDTA ----------------------------------------------------------------------
def to_deepsmiles(smiles: str):
    converter = deepsmiles.Converter(rings=True, branches=True)
    deep_smiles = converter.encode(smiles)

    return deep_smiles


def seq_to_words(sequence: str, word_len: int):

    words = ()
    sequence_length = len(sequence)
    for start_index in range(word_len):
        for i in range(start_index, sequence_length, word_len):
            substring = sequence[i : i + word_len]
            if len(substring) == word_len:
                words += (substring,)

    return words


# ------------------------------------------------------------------------------


def one_hot_encode(x, allowable_set) -> np.array:
    if x not in allowable_set:
        logging.warning(f"Input {x} not in allowable set {allowable_set}.")
        return np.zeros(len(allowable_set), dtype=int)

    return np.array([x == s for s in allowable_set], dtype=int)


# Original
def onehot(x):
    one_d = {}
    ps1 = list(x.values())
    p_set = set()
    lens_p = [len(p) for p in ps1]
    for p in ps1:
        p_set = p_set.union(set(p))
    char_to_int_p = dict((c, i) for i, c in enumerate(p_set))
    int_to_char_p = dict((i, c) for i, c in enumerate(p_set))
    # onehot_p = np.zeros((len(ps1), len(char_to_int_p), max(lens_p)))
    for i, p in enumerate(ps1):
        onehot_p = np.zeros((len(char_to_int_p), max(lens_p)))
        for j, char in enumerate(p):
            onehot_p[char_to_int_p[char], j] = 1.0
        one_d[i] = onehot_p
    return one_d


def onehot_words(x):
    one_d = {}
    sequences = list(x.values())
    unique_words = set()
    for sequence in sequences:
        unique_words = unique_words.union(set(sequence.split()))
    word_to_int = {word: i for i, word in enumerate(unique_words)}

    for i, sequence in enumerate(sequences):
        words = sequence.split()
        onehot_sequence = np.zeros((len(words), len(word_to_int)))
        for j, word in enumerate(words):
            if (
                word in word_to_int
            ):  # Check if the word is in the dictionary to handle out-of-vocabulary words
                onehot_sequence[j, word_to_int[word]] = 1.0
        one_d[i] = onehot_sequence
    return one_d

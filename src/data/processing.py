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


def seq_to_words(sequence: str, word_len: int, max_length: int):
    words = ()
    sequence_length = len(sequence)
    count = 0

    for start_index in range(word_len):
        for i in range(start_index, sequence_length, word_len):
            if count >= max_length:
                return words
            substring = sequence[i : i + word_len]
            if len(substring) == word_len:
                words += (substring,)
                count += 1

    return words


def make_words_set(sequences: list[tuple]):
    words_set = set(word for seq in sequences for word in seq)

    return words_set


# ------------------------------------------------------------------------------
def one_hot_encode(x, allowable_set) -> np.array:
    if x not in allowable_set:
        logging.warning(f"Input {x} not in allowable set {allowable_set}.")
        return np.zeros(len(allowable_set), dtype=int)

    return np.array([x == s for s in allowable_set], dtype=int)


def one_hot_words(x, allowable_set, length: int):
    word_to_int = {word: i + 1 for i, word in enumerate(allowable_set)}
    indices_sequence = np.zeros(length, dtype=int)

    for idx, word in enumerate(x):
        if word in word_to_int:
            indices_sequence[idx] = word_to_int[word]
        else:
            indices_sequence[idx] = 0.0

    return indices_sequence

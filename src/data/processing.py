"""
Functions for labeling and encoding chemical characters like Compound SMILES and atom string, refer to
https://github.com/hkmztrk/DeepDTA and https://github.com/thinng/GraphDTA.
"""

import logging

import numpy as np
from rdkit import Chem
import networkx as nx

from .constants import Tokens, AtomFeatures


# Functions --------------------------------------------------------------------
def one_hot_encode(x, allowable_set):
    if x not in allowable_set:
        logging.warning(f"Input {x} not in allowable set {allowable_set}.")
        return np.zeros(len(allowable_set), dtype=int)

    return np.array([x == s for s in allowable_set], dtype=int)


def one_hot_encode_with_unknown(x, allowable_set):
    x = x if x in allowable_set else allowable_set[-1]

    return np.array([x == s for s in allowable_set], dtype=int)


def get_atom_features(atom):
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


def smile_to_graph(smiles):
    """ """
    mol = Chem.MolFromSmiles(smiles)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        atom_features = get_atom_features(atom)
        features.append(atom_features / sum(atom_features))

    edges = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]

    di_graph = nx.Graph(edges).to_directed()
    edge_index = list(di_graph.edges)

    return c_size, features, edge_index


# -------------------------------------------------------------------------------
# TODO: Fix style
def tokenize_smiles(smiles, max_length=85, isomeric=False):
    """ """
    if not isomeric:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.warning(f"rdkit cannot find this SMILES {smiles}.")
        else:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(smiles[:max_length]):
        encoding[idx] = Tokens.CHARISOSMISET.get(letter, 0)
        if encoding[idx] == 0:
            logging.warning(
                f"Character '{letter}' not found in SMILES set, treated as padding."
            )

    return encoding


def tokenize_target(sequence, max_length=1200):
    """ """

    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        letter = letter.upper()
        encoding[idx] = Tokens.CHARPROTSET.get(letter, 0)
        if encoding[idx] == 0:
            logging.warning(
                f"Character '{letter}' not found in protein set, treated as padding."
            )
    return encoding

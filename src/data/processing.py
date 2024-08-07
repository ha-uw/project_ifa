"""
Functions for labeling and encoding chemical characters like Compound SMILES and atom string, refer to
https://github.com/hkmztrk/DeepDTA and https://github.com/thinng/GraphDTA.
"""

import logging

import numpy as np
from rdkit import Chem
from functools import lru_cache
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


@lru_cache(maxsize=32)
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


@lru_cache(maxsize=32)
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


@lru_cache(maxsize=32)
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


@lru_cache(maxsize=32)
def tokenize_target(sequence: str, max_length: int = 1200) -> np.array:
    """Tokenizes a protein sequence."""

    return tokenize_sequence(sequence.upper(), Tokens.CHARPROTSET, max_length)


# WideDTA ----------------------------------------------------------------------
@lru_cache(maxsize=32)
def to_deepsmiles(smiles: str):
    converter = deepsmiles.Converter(rings=True, branches=True)
    deep_smiles = converter.encode(smiles)

    return deep_smiles


# def seq_to_words(sequence: str, word_len: int, max_length: int):
#     words = ()
#     sequence_length = len(sequence)
#     count = 0

#     for start_index in range(word_len):
#         for i in range(start_index, sequence_length, word_len):
#             if count >= max_length:
#                 return words
#             substring = sequence[i : i + word_len]
#             if len(substring) == word_len:
#                 words += (substring,)
#                 count += 1

#     return words


# def seq_to_words(sequence: str, word_len: int, max_length: int):
#     sequence_array = np.array(list(sequence))

#     words = []

#     # Iterate over each possible starting index within the word length
#     for start_index in range(word_len):
#         # Calculate the end index for slicing by stepping word_len at a time
#         end_index = sequence_array.size

#         # Slice the array from start_index to the end, stepping by word_len
#         sliced_words = sequence_array[start_index:end_index:word_len]

#         # Calculate how many full words we can take from this slice
#         num_full_words = min(
#             len(sliced_words) * word_len // word_len, max_length - len(words)
#         )

#         # Convert sliced words back to strings and add to the words list
#         for i in range(num_full_words):
#             word = "".join(sliced_words[i * word_len : (i + 1) * word_len])
#             words.append(word)
#             if len(words) >= max_length:
#                 break

#         # If we've reached the max_length, stop processing
#         if len(words) >= max_length:
#             break

#     # Convert the list of words back to a tuple before returning
#     return tuple(words)


@lru_cache(maxsize=32)
def seq_to_words(sequence: str, word_len: int, max_length: int):
    # Early exit for invalid input
    if word_len <= 0 or max_length <= 0:
        return ()

    words = []
    sequence_length = len(sequence)
    # Calculate the total possible words to be extracted
    total_possible_words = sum(
        (sequence_length - start_index) // word_len for start_index in range(word_len)
    )
    # Iterate up to the minimum of max_length and total_possible_words
    for start_index in range(word_len):
        for i in range(start_index, sequence_length, word_len):
            if len(words) >= min(max_length, total_possible_words):
                return tuple(words)

            substring = sequence[i : i + word_len]
            if len(substring) == word_len:
                words.append(substring)

    return tuple(words)


def make_words_dict(sequences):
    words_set = set(word for seq in sequences for word in seq)
    word_to_int = {word: i for i, word in enumerate(words_set, start=1)}

    return word_to_int


def encode_word(x, word_to_int, length: int) -> np.array:
    indices_sequence = np.zeros(length, dtype=int)

    # Limit the loop to the minimum of the length of x and the specified length
    for idx in range(min(len(x), length)):
        word = x[idx]
        indices_sequence[idx] = word_to_int.get(word, 0)

    return indices_sequence


# ------------------------------------------------------------------------------


# def encode_word(x, allowable_set, length: int) -> np.array:
#     word_to_int = {word: i for i, word in enumerate(allowable_set, start=1)}
#     indices_sequence = np.zeros(length, dtype=int)

#     # Limit the loop to the minimum of the length of x and the specified length
#     for idx in range(min(len(x), length)):
#         word = x[idx]
#         indices_sequence[idx] = word_to_int.get(word, 0)

#     return indices_sequence


# def encode_word(x, allowable_set, length: int) -> np.array:
#     word_to_int = {word: i + 1 for i, word in enumerate(allowable_set)}
#     indices_sequence = np.zeros(length, dtype=int)

#     for idx, word in enumerate(x):
#         if word in word_to_int:
#             indices_sequence[idx] = word_to_int[word]
#         else:
#             indices_sequence[idx] = 0

#     return indices_sequence

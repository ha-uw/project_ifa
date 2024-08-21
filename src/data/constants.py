"""
constants.py

This module contains dictionaries used throughout the application.
"""


class Tokens:
    """
    Class representing a collection of tokens.

    Attributes:
        CHARPROTSET (dict): A dictionary mapping protein sequence characters (amino acids) to integers.
        CHARISOSMISET (dict): A dictionary mapping SMILES characters to integers.
    """

    CHARPROTSET = {
        "A": 1,
        "C": 2,
        "B": 3,
        "E": 4,
        "D": 5,
        "G": 6,
        "F": 7,
        "I": 8,
        "H": 9,
        "K": 10,
        "M": 11,
        "L": 12,
        "O": 13,
        "N": 14,
        "Q": 15,
        "P": 16,
        "S": 17,
        "R": 18,
        "U": 19,
        "T": 20,
        "W": 21,
        "V": 22,
        "Y": 23,
        "X": 24,
        "Z": 25,
    }

    CHARISOSMISET = {
        "#": 29,
        "%": 30,
        ")": 31,
        "(": 1,
        "+": 32,
        "-": 33,
        "/": 34,
        ".": 2,
        "1": 35,
        "0": 3,
        "3": 36,
        "2": 4,
        "5": 37,
        "4": 5,
        "7": 38,
        "6": 6,
        "9": 39,
        "8": 7,
        "=": 40,
        "A": 41,
        "@": 8,
        "C": 42,
        "B": 9,
        "E": 43,
        "D": 10,
        "G": 44,
        "F": 11,
        "I": 45,
        "H": 12,
        "K": 46,
        "M": 47,
        "L": 13,
        "O": 48,
        "N": 14,
        "P": 15,
        "S": 49,
        "R": 16,
        "U": 50,
        "T": 17,
        "W": 51,
        "V": 18,
        "Y": 52,
        "[": 53,
        "Z": 19,
        "]": 54,
        "\\": 20,
        "a": 55,
        "c": 56,
        "b": 21,
        "e": 57,
        "d": 22,
        "g": 58,
        "f": 23,
        "i": 59,
        "h": 24,
        "m": 60,
        "l": 25,
        "o": 61,
        "n": 26,
        "s": 62,
        "r": 27,
        "u": 63,
        "t": 28,
        "y": 64,
    }


class AtomFeatures:
    """
    Class representing atom features.
    Attributes:
        ALLOWED_VALUES (list): List of allowed values for atom features.
        CHARATOMSET (list): List of character atom sets.
    """

    ALLOWED_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    CHARATOMSET = [
        "C",
        "N",
        "O",
        "S",
        "F",
        "Si",
        "P",
        "Cl",
        "Br",
        "Mg",
        "Na",
        "Ca",
        "Fe",
        "As",
        "Al",
        "I",
        "B",
        "V",
        "K",
        "Tl",
        "Yb",
        "Sb",
        "Sn",
        "Ag",
        "Pd",
        "Co",
        "Se",
        "Ti",
        "Zn",
        "H",
        "Li",
        "Ge",
        "Cu",
        "Au",
        "Ni",
        "Cd",
        "In",
        "Mn",
        "Zr",
        "Cr",
        "Pt",
        "Hg",
        "Pb",
        "Unknown",
    ]

"""
ismip6_regions.py — ISMIP6 basin letter ↔ geographic name mapping.

Shared by all regional scripts so names are consistent across figures.
"""
from __future__ import annotations

# ISMIP6 letter → geographic name (from Rignot et al. 2013 / Paolo 2023)
LETTER_TO_NAME: dict[str, str] = {
    "A-Ap":  "Dronning Maud Land",
    "Ap-B":  "Enderby Land",
    "B-C":   "Amery–Lambert",
    "C-Cp":  "Philippi–Denman",
    "Cp-D":  "Totten",
    "D-Dp":  "Mertz",
    "Dp-E":  "Victoria Land",
    "E-F":   "Ross",
    "F-G":   "Getz",
    "G-H":   "Thwaites/PIG",
    "H-Hp":  "Bellingshausen",
    "Hp-I":  "George VI",
    "I-Ipp": "Larsen A–C",
    "Ipp-J": "Larsen E",
    "J-K":   "FRIS",
    "K-A":   "Brunt–Stancomb",
}

# Ordered list (index i = basin i in the 16-region mask)
BASIN_NAMES: list[str] = [LETTER_TO_NAME[k] for k in [
    "A-Ap", "Ap-B", "B-C", "C-Cp", "Cp-D", "D-Dp", "Dp-E", "E-F",
    "F-G", "G-H", "H-Hp", "Hp-I", "I-Ipp", "Ipp-J", "J-K", "K-A",
]]

# Short labels for tight subplots
SHORT_LABELS: dict[str, str] = {
    "Dronning Maud Land": "DML",
    "Enderby Land": "Enderby",
    "Amery–Lambert": "Amery",
    "Philippi–Denman": "Philippi",
    "Totten": "Totten",
    "Mertz": "Mertz",
    "Victoria Land": "Vic. Land",
    "Ross": "Ross",
    "Getz": "Getz",
    "Thwaites/PIG": "Thwaites/PIG",
    "Bellingshausen": "Belling.",
    "George VI": "George VI",
    "Larsen A–C": "Larsen A–C",
    "Larsen E": "Larsen E",
    "FRIS": "FRIS",
    "Brunt–Stancomb": "Brunt",
}


def basin_name(index: int) -> str:
    """Return geographic name for ISMIP6 basin index (0–15)."""
    if 0 <= index < len(BASIN_NAMES):
        return BASIN_NAMES[index]
    return f"basin_{index}"


def short_label(index: int) -> str:
    """Return short label for ISMIP6 basin index (0–15)."""
    name = basin_name(index)
    return SHORT_LABELS.get(name, name)


def letter_from_index(index: int) -> str:
    """Return ISMIP6 letter label for basin index (0–15), e.g. 'A-Ap'."""
    letters = [
        "A-Ap", "Ap-B", "B-C", "C-Cp", "Cp-D", "D-Dp", "Dp-E", "E-F",
        "F-G", "G-H", "H-Hp", "Hp-I", "I-Ipp", "Ipp-J", "J-K", "K-A",
    ]
    if 0 <= index < len(letters):
        return letters[index]
    return f"?{index}"

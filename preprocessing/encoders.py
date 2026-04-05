"""
glassbox.preprocessing.encoders
=================================

**OneHotEncoder** and **LabelEncoder** — encode categorical features.

Both encoders
-------------
* Target columns whose ``inferred_type`` is ``"categorical"`` (or ``"boolean"``
  for LabelEncoder, which is configurable via ``target_types``).
* Leave non-targeted columns untouched.
* Return a **copy** of the data — the original array is never modified.
* Implement ``fit()``, ``transform()``, and ``fit_transform()``.

OneHotEncoder  (nominal data)
------------------------------
Creates one binary indicator column per unique category, removing the
original column:

.. code-block::

    dept = ["Eng", "HR", "Eng"]
    → dept_Eng = [1, 0, 1],  dept_HR = [0, 1, 0]

Categories are sorted lexicographically for deterministic output.
Optionally the first category can be dropped (``drop_first=True``) to avoid
the dummy-variable trap in linear models.

LabelEncoder  (ordinal data)
------------------------------
Maps each unique category to a non-negative integer (0-based, sorted
lexicographically for reproducibility):

.. code-block::

    size = ["S", "M", "L", "XL"]
    → size = [3, 1, 0, 2]   (alphabetical: L=0, M=1, S=2, XL=3)

The mapping is stored per column and can be inspected via ``get_mapping()``.

Public API
----------
OneHotEncoder
LabelEncoder
EncoderSummary  — per-column record of categories found and encoding applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from glassbox.eda.auto_typer import AutoTyper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a"}


def _is_missing(val) -> bool:
    """Return True if *val* represents a missing value."""
    if val is None:
        return True
    try:
        if np.isnan(float(val)):
            return True
    except (ValueError, TypeError):
        pass
    if isinstance(val, str) and val.strip().lower() in _MISSING_STRINGS:
        return True
    return False


def _missing_mask(col: np.ndarray) -> np.ndarray:
    """Boolean mask of missing entries in *col*."""
    if col.dtype.kind == "f":
        return np.isnan(col)
    return np.array([_is_missing(v) for v in col], dtype=bool)


def _validate_input(
    data: np.ndarray,
    headers: Optional[List[str]],
) -> Tuple[np.ndarray, List[str]]:
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_cols = data.shape[1]
    if headers is None:
        headers = [f"col_{i}" for i in range(n_cols)]
    if len(headers) != n_cols:
        raise ValueError(
            f"Length of headers ({len(headers)}) does not match "
            f"number of columns ({n_cols})."
        )
    return data, headers


def _get_type_map(
    data: np.ndarray,
    headers: List[str],
    type_map: Optional[Dict[str, str]],
) -> Dict[str, str]:
    if type_map is not None:
        return type_map
    typer = AutoTyper()
    type_infos = typer.detect(data, headers)
    return {ti.name: ti.inferred_type for ti in type_infos}


def _unique_sorted(col: np.ndarray) -> List[str]:
    """Return sorted unique non-missing string values from *col*."""
    seen: Set[str] = set()
    for v in col:
        if not _is_missing(v):
            seen.add(str(v))
    return sorted(seen)


# ---------------------------------------------------------------------------
# Result container (shared by both encoders)
# ---------------------------------------------------------------------------
@dataclass
class EncoderSummary:
    """Records encoding details for a single column.

    Attributes
    ----------
    column : str
        Original column name.
    encoder_type : str
        ``"onehot"`` or ``"label"``.
    categories : list[str]
        Unique category values found during ``fit()`` (sorted).
    output_columns : list[str]
        Names of the columns produced (same as input for label encoding;
        new indicator names for one-hot encoding).
    mapping : dict[str, int] | None
        Integer mapping per category (LabelEncoder only).
    """

    column: str
    encoder_type: str
    categories: List[str] = field(default_factory=list)
    output_columns: List[str] = field(default_factory=list)
    mapping: Optional[Dict[str, int]] = None

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dictionary."""
        d: dict = {
            "column": self.column,
            "encoder_type": self.encoder_type,
            "categories": self.categories,
            "output_columns": self.output_columns,
        }
        if self.mapping is not None:
            d["mapping"] = self.mapping
        return d


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------
class OneHotEncoder:
    """Encode nominal categorical columns as binary indicator columns.

    For each target column a new binary column is created for every unique
    category observed during ``fit()``.  The original categorical column is
    removed and replaced by the indicator columns.

    Column naming convention: ``<original_col>_<category>``.

    Unknown categories encountered during ``transform()`` (i.e. values not
    seen during ``fit()``) are represented as all-zero indicator rows.

    Parameters
    ----------
    drop_first : bool, optional
        Drop the first (alphabetically) indicator column per original column
        to avoid multicollinearity in linear models.  Default ``False``.
    target_types : set[str], optional
        Column types to encode.  Default ``{"categorical"}``.

    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.preprocessing.encoders import OneHotEncoder
    >>> data = np.array([["Eng"], ["HR"], ["Eng"]], dtype=object)
    >>> enc = OneHotEncoder()
    >>> out_data, out_headers, summaries = enc.fit_transform(data, headers=["dept"])
    >>> out_headers
    ['dept_Eng', 'dept_HR']
    >>> out_data
    array([[1, 0],
           [0, 1],
           [1, 0]])
    """

    def __init__(
        self,
        drop_first: bool = False,
        target_types: Optional[Set[str]] = None,
    ) -> None:
        self.drop_first = drop_first
        self.target_types = target_types or {"categorical"}

        # State set during fit()
        self._categories: Dict[str, List[str]] = {}  # col_name → sorted categories
        self._target_cols: List[str] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
        type_map: Optional[Dict[str, str]] = None,
    ) -> "OneHotEncoder":
        """Learn the unique categories for each target column.

        Parameters
        ----------
        data : np.ndarray
            2-D training data.
        headers : list[str], optional
            Column names.
        type_map : dict[str, str], optional
            ``{col_name: inferred_type}``.  Auto-detected when *None*.

        Returns
        -------
        self
        """
        data, headers = _validate_input(data, headers)
        type_map = _get_type_map(data, headers, type_map)

        self._categories = {}
        self._target_cols = []

        for i, col_name in enumerate(headers):
            if type_map.get(col_name) not in self.target_types:
                continue
            cats = _unique_sorted(data[:, i])
            if len(cats) == 0:
                continue
            self._categories[col_name] = cats
            self._target_cols.append(col_name)

        self._fitted = True
        return self

    def transform(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str], List[EncoderSummary]]:
        """Replace categorical columns with binary indicator columns.

        Parameters
        ----------
        data : np.ndarray
            2-D data to encode.
        headers : list[str], optional
            Column names (must match those seen in ``fit()``).

        Returns
        -------
        tuple[np.ndarray, list[str], list[EncoderSummary]]
            Encoded data array, new column names, and per-column summaries.
            The returned array may have *more* columns than the input.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        data, headers = _validate_input(data, headers)

        n_rows = data.shape[0]
        new_cols: List[np.ndarray] = []
        new_headers: List[str] = []
        summaries: List[EncoderSummary] = []

        for i, col_name in enumerate(headers):
            if col_name not in self._categories:
                # Pass-through: keep the original column unchanged.
                new_cols.append(data[:, i].reshape(-1, 1))
                new_headers.append(col_name)
                continue

            cats = self._categories[col_name]
            col = data[:, i]
            missing = _missing_mask(col)

            # Determine which categories to output (drop_first skips cats[0]).
            output_cats = cats[1:] if self.drop_first else cats
            output_col_names = [f"{col_name}_{c}" for c in output_cats]

            indicator_block = np.zeros((n_rows, len(output_cats)), dtype=np.int8)
            for j, cat in enumerate(output_cats):
                indicator_block[:, j] = np.array(
                    [(not missing[r]) and (str(col[r]) == cat) for r in range(n_rows)],
                    dtype=np.int8,
                )

            new_cols.append(indicator_block)
            new_headers.extend(output_col_names)

            summaries.append(EncoderSummary(
                column=col_name,
                encoder_type="onehot",
                categories=cats,
                output_columns=output_col_names,
            ))

        # Concatenate along axis-1 into a single object array.
        result = np.concatenate(
            [c.reshape(n_rows, -1) for c in new_cols], axis=1
        ).astype(object)

        return result, new_headers, summaries

    def fit_transform(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
        type_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[np.ndarray, List[str], List[EncoderSummary]]:
        """Fit on *data* then transform it in one step."""
        return self.fit(data, headers, type_map).transform(data, headers)

    def get_feature_names(self) -> List[str]:
        """Return all indicator column names that would be produced for each fitted column."""
        names: List[str] = []
        for col_name in self._target_cols:
            cats = self._categories.get(col_name, [])
            output_cats = cats[1:] if self.drop_first else cats
            names.extend(f"{col_name}_{c}" for c in output_cats)
        return names


# ---------------------------------------------------------------------------
# LabelEncoder
# ---------------------------------------------------------------------------
class LabelEncoder:
    """Encode ordinal categorical columns as non-negative integers.

    Each unique category is mapped to an integer in ``[0, n_categories - 1]``
    based on alphabetical (lexicographic) ordering, which ensures the mapping
    is deterministic and reproducible.

    Unknown categories encountered during ``transform()`` are mapped to
    ``-1`` to make them distinguishable.

    Parameters
    ----------
    target_types : set[str], optional
        Column types to encode.  Default ``{"categorical"}``.

    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.preprocessing.encoders import LabelEncoder
    >>> data = np.array([["S"], ["M"], ["L"], ["XL"]], dtype=object)
    >>> enc = LabelEncoder()
    >>> out_data, summaries = enc.fit_transform(data, headers=["size"])
    >>> out_data[:, 0]
    array([2, 1, 0, 3], dtype=object)   # L=0, M=1, S=2, XL=3
    """

    def __init__(self, target_types: Optional[Set[str]] = None) -> None:
        self.target_types = target_types or {"categorical"}

        # State set during fit()
        self._mapping: Dict[str, Dict[str, int]] = {}  # col_name → {cat: int}
        self._categories: Dict[str, List[str]] = {}
        self._target_cols: List[str] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
        type_map: Optional[Dict[str, str]] = None,
    ) -> "LabelEncoder":
        """Learn the integer mapping for each target column.

        Parameters
        ----------
        data : np.ndarray
            2-D training data.
        headers : list[str], optional
            Column names.
        type_map : dict[str, str], optional
            ``{col_name: inferred_type}``.  Auto-detected when *None*.

        Returns
        -------
        self
        """
        data, headers = _validate_input(data, headers)
        type_map = _get_type_map(data, headers, type_map)

        self._mapping = {}
        self._categories = {}
        self._target_cols = []

        for i, col_name in enumerate(headers):
            if type_map.get(col_name) not in self.target_types:
                continue
            cats = _unique_sorted(data[:, i])
            if len(cats) == 0:
                continue
            self._categories[col_name] = cats
            self._mapping[col_name] = {cat: idx for idx, cat in enumerate(cats)}
            self._target_cols.append(col_name)

        self._fitted = True
        return self

    def transform(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[EncoderSummary]]:
        """Map categorical values to integers in-place (on a copy).

        Parameters
        ----------
        data : np.ndarray
            2-D data to encode.
        headers : list[str], optional
            Column names (must match those seen in ``fit()``).

        Returns
        -------
        tuple[np.ndarray, list[EncoderSummary]]
            A *copy* of *data* with target columns replaced by integers, and a
            per-column summary of the mapping applied.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        data, headers = _validate_input(data, headers)

        result = data.copy().astype(object)
        summaries: List[EncoderSummary] = []

        for i, col_name in enumerate(headers):
            if col_name not in self._mapping:
                continue
            mapping = self._mapping[col_name]
            col = result[:, i]

            encoded = np.array(
                [mapping.get(str(v), -1) if not _is_missing(v) else np.nan
                 for v in col],
                dtype=object,
            )
            result[:, i] = encoded

            summaries.append(EncoderSummary(
                column=col_name,
                encoder_type="label",
                categories=self._categories[col_name],
                output_columns=[col_name],
                mapping=mapping,
            ))

        return result, summaries

    def fit_transform(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
        type_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[np.ndarray, List[EncoderSummary]]:
        """Fit on *data* then transform it in one step."""
        return self.fit(data, headers, type_map).transform(data, headers)

    def get_mapping(self, col_name: str) -> Optional[Dict[str, int]]:
        """Return the integer mapping for *col_name* (or ``None`` if not fitted)."""
        return self._mapping.get(col_name)

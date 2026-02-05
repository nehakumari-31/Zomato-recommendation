"""
Phase 2: User Input validation

Validates and normalizes:
- city (against dataset city list)
- cuisine (against dataset cuisine list)
- price (exact, range, or category) for approx cost for two
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence, Tuple
import difflib
import re


class ValidationError(ValueError):
    """Raised when user input cannot be validated."""


def normalize_token(value: str) -> str:
    """
    Normalize a user-provided token for matching.
    - trim whitespace
    - collapse internal spaces
    - lowercase
    """
    if value is None:
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


class PriceCategory(str, Enum):
    BUDGET = "budget"
    MODERATE = "moderate"
    PREMIUM = "premium"


@dataclass(frozen=True)
class PricePreference:
    """
    Represents user price preference for cost-for-two.

    Exactly one of:
    - exact: a single numeric target
    - range: min/max inclusive
    - category: budget/moderate/premium mapped to a range
    """

    exact: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    category: Optional[PriceCategory] = None

    def as_range(self, *, default_tolerance_pct: float = 0.10, default_tolerance_abs: float = 100.0) -> Tuple[Optional[float], Optional[float]]:
        """
        Return (min, max) range representation.

        - If range provided, returns it.
        - If exact provided, returns a tolerance window around it.
        - If category provided, returns category range.
        """
        if self.min_value is not None or self.max_value is not None:
            return (self.min_value, self.max_value)

        if self.exact is not None:
            tol = max(default_tolerance_abs, self.exact * default_tolerance_pct)
            return (max(0.0, self.exact - tol), self.exact + tol)

        return (None, None)


@dataclass(frozen=True)
class ValidatedUserInput:
    """
    STEP 2 output object.

    - city/cuisine are returned as canonical strings from dataset lists
    - price is returned as a normalized PricePreference (exact/range/category)
    """

    city: str
    cuisine: str
    price: PricePreference


def _best_close_matches(query: str, choices: Sequence[str], *, n: int = 5, cutoff: float = 0.6) -> Sequence[str]:
    return difflib.get_close_matches(query, choices, n=n, cutoff=cutoff)


def validate_city(user_city: str, available_cities: Sequence[str]) -> str:
    """
    Validate city against a list of dataset cities.

    Matching is case-insensitive and whitespace-insensitive.
    Returns the canonical city string from available_cities.
    """
    if not available_cities:
        raise ValidationError("No cities available for validation.")

    q = normalize_token(user_city)
    if not q:
        raise ValidationError("City is required.")

    # Build normalization map
    norm_map = {normalize_token(c): c for c in available_cities if c is not None}

    if q in norm_map:
        return norm_map[q]

    # Try partial containment (e.g., user types "bang" for "Bangalore")
    contains = [c for nrm, c in norm_map.items() if q in nrm]
    if len(contains) == 1:
        return contains[0]

    suggestions = _best_close_matches(q, list(norm_map.keys()))
    if suggestions:
        suggested = [norm_map[s] for s in suggestions]
        raise ValidationError(f"City not found. Did you mean: {', '.join(suggested[:5])}?")

    raise ValidationError("City not found.")


def validate_cuisine(user_cuisine: str, available_cuisines: Sequence[str]) -> str:
    """
    Validate cuisine against a list of dataset cuisines.

    Returns the canonical cuisine string from available_cuisines.
    """
    if not available_cuisines:
        raise ValidationError("No cuisines available for validation.")

    q = normalize_token(user_cuisine)
    if not q:
        raise ValidationError("Cuisine is required.")

    norm_map = {normalize_token(c): c for c in available_cuisines if c is not None}

    if q in norm_map:
        return norm_map[q]

    # Allow partial match for convenience
    contains = [c for nrm, c in norm_map.items() if q in nrm]
    if len(contains) == 1:
        return contains[0]

    suggestions = _best_close_matches(q, list(norm_map.keys()))
    if suggestions:
        suggested = [norm_map[s] for s in suggestions]
        raise ValidationError(f"Cuisine not found. Did you mean: {', '.join(suggested[:5])}?")

    raise ValidationError("Cuisine not found.")


def parse_price_preference(
    user_price: str,
    *,
    budget_max: float = 500.0,
    moderate_max: float = 1000.0,
) -> PricePreference:
    """
    Parse price input.

    Accepted formats:
    - exact: "500", "₹500", "500.0"
    - range: "500-1000", "500 to 1000", "500–1000"
    - category: "budget", "moderate", "premium"
    """
    q = normalize_token(user_price)
    if not q:
        raise ValidationError("Price is required.")

    # Categories
    if q in ("budget", "low", "cheap"):
        return PricePreference(min_value=0.0, max_value=budget_max, category=PriceCategory.BUDGET)
    if q in ("moderate", "mid", "medium"):
        return PricePreference(min_value=budget_max, max_value=moderate_max, category=PriceCategory.MODERATE)
    if q in ("premium", "high", "expensive"):
        return PricePreference(min_value=moderate_max, max_value=None, category=PriceCategory.PREMIUM)

    # Normalize range separators
    q2 = q.replace("–", "-").replace("—", "-")
    q2 = q2.replace(" to ", "-")

    # Extract numbers (supports decimals)
    nums = re.findall(r"\d+(?:\.\d+)?", q2)
    if not nums:
        raise ValidationError("Price must be a number, range, or category (budget/moderate/premium).")

    if len(nums) == 1:
        return PricePreference(exact=float(nums[0]))

    # Take first two as min/max
    mn = float(nums[0])
    mx = float(nums[1])
    if mn > mx:
        mn, mx = mx, mn
    return PricePreference(min_value=mn, max_value=mx)


def validate_user_input(
    *,
    city: str,
    cuisine: str,
    price: str,
    available_cities: Sequence[str],
    available_cuisines: Sequence[str],
    budget_max: float = 500.0,
    moderate_max: float = 1000.0,
) -> ValidatedUserInput:
    """
    Convenience wrapper to validate all Step 2 inputs together.

    Returns the Step 2 output: validated (city, cuisine, price).
    """
    v_city = validate_city(city, available_cities)
    v_cuisine = validate_cuisine(cuisine, available_cuisines)
    v_price = parse_price_preference(price, budget_max=budget_max, moderate_max=moderate_max)
    return ValidatedUserInput(city=v_city, cuisine=v_cuisine, price=v_price)


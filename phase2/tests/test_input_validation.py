import pytest

from phase2.input_validation import (
    ValidationError,
    PriceCategory,
    parse_price_preference,
    validate_user_input,
    validate_city,
    validate_cuisine,
)


def test_validate_city_exact_case_insensitive():
    cities = ["Bangalore", "Mumbai", "Delhi"]
    assert validate_city("bangalore", cities) == "Bangalore"
    assert validate_city("  Mumbai  ", cities) == "Mumbai"


def test_validate_city_partial_unique():
    cities = ["Bangalore", "Mumbai", "Delhi"]
    assert validate_city("del", cities) == "Delhi"


def test_validate_city_suggests():
    cities = ["Bangalore", "Mumbai", "Delhi"]
    with pytest.raises(ValidationError) as e:
        validate_city("Bangaloru", cities)
    assert "Did you mean" in str(e.value)


def test_validate_city_required():
    with pytest.raises(ValidationError):
        validate_city(" ", ["Bangalore"])


def test_validate_cuisine_exact_and_partial():
    cuisines = ["North Indian", "South Indian", "Chinese", "Italian"]
    assert validate_cuisine("north indian", cuisines) == "North Indian"
    assert validate_cuisine("itali", cuisines) == "Italian"


def test_validate_cuisine_suggests():
    cuisines = ["North Indian", "South Indian", "Chinese", "Italian"]
    with pytest.raises(ValidationError) as e:
        validate_cuisine("Chineese", cuisines)
    assert "Did you mean" in str(e.value)


def test_parse_price_exact_number():
    pref = parse_price_preference("500")
    assert pref.exact == 500.0
    assert pref.min_value is None and pref.max_value is None


def test_parse_price_exact_with_currency():
    pref = parse_price_preference("₹750")
    assert pref.exact == 750.0


def test_parse_price_range_hyphen():
    pref = parse_price_preference("500-1000")
    assert pref.min_value == 500.0
    assert pref.max_value == 1000.0


def test_parse_price_range_to():
    pref = parse_price_preference("500 to 1000")
    assert pref.min_value == 500.0
    assert pref.max_value == 1000.0


def test_parse_price_range_swapped():
    pref = parse_price_preference("1000-500")
    assert pref.min_value == 500.0
    assert pref.max_value == 1000.0


def test_parse_price_category_budget():
    pref = parse_price_preference("budget", budget_max=500.0, moderate_max=1000.0)
    assert pref.category == PriceCategory.BUDGET
    assert pref.min_value == 0.0
    assert pref.max_value == 500.0


def test_parse_price_category_moderate():
    pref = parse_price_preference("moderate", budget_max=500.0, moderate_max=1000.0)
    assert pref.category == PriceCategory.MODERATE
    assert pref.min_value == 500.0
    assert pref.max_value == 1000.0


def test_parse_price_category_premium():
    pref = parse_price_preference("premium", budget_max=500.0, moderate_max=1000.0)
    assert pref.category == PriceCategory.PREMIUM
    assert pref.min_value == 1000.0
    assert pref.max_value is None


def test_parse_price_required():
    with pytest.raises(ValidationError):
        parse_price_preference(" ")


def test_parse_price_invalid():
    with pytest.raises(ValidationError):
        parse_price_preference("whatever")


def test_validate_user_input_happy_path():
    cities = ["Bangalore", "Mumbai", "Delhi"]
    cuisines = ["North Indian", "Chinese", "Italian"]
    result = validate_user_input(
        city=" bangalore ",
        cuisine="chinese",
        price="500-1000",
        available_cities=cities,
        available_cuisines=cuisines,
    )
    assert result.city == "Bangalore"
    assert result.cuisine == "Chinese"
    assert result.price.min_value == 500.0
    assert result.price.max_value == 1000.0


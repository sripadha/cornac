from cart import add_item, describe, total


def test_add_item_appends_to_given_list():
    assert add_item("pear", ["apple"]) == ["apple", "pear"]


def test_add_item_starts_fresh_each_call():
    assert add_item("apple") == ["apple"]
    assert add_item("pear") == ["pear"]


def test_total_and_describe():
    assert total([1.10, 2.25]) == 3.35
    assert describe([]) == "(empty)"
    assert describe(["apple", "pear"]) == "apple, pear"

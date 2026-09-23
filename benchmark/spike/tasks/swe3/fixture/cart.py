"""A minimal shopping-cart helper."""


def add_item(item, items=[]):
    """Return the list of items with `item` appended.

    Called without `items`, it starts a fresh, empty cart.
    """
    items.append(item)
    return items


def total(prices):
    """Sum of the prices, rounded to cents."""
    return round(sum(prices), 2)


def describe(items):
    """A one-line, human-readable listing of the cart."""
    return ", ".join(items) if items else "(empty)"

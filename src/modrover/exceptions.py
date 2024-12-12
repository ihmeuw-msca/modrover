class NotFittedError(Exception):
    """Raised if a user tries to call predict on an unfitted learner/rover."""


class InvalidConfigurationError(Exception):
    """Raised if a provided configuration parameter violates a rule."""

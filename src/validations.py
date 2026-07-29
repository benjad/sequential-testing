def validate_probability(value: float) -> bool:
    if not (0 < value < 1):
        raise ValueError("Input value should be between 0 and 1")


def validate_lenght(x: list, y: list) -> bool:
    if len(x) != len(y):
        raise ValueError("X and Y lenghts should be the same")


def validate_options(user_input: str, options: list[str]):
    if user_input not in options:
        raise ValueError(f"'{user_input}' is not a valid choice. Choose from {options}.")

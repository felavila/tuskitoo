

def format_scientific(value, precision=3):
    """
    Format a value in scientific notation with a specified number of decimal places.
    
    Parameters:
        value (float): The value to be formatted.
        precision (int): The number of decimal places for the coefficient.

    Returns:
        str: The formatted value in scientific notation.
    """
    coefficient, exponent = "{:.{precision}e}".format(value, precision=precision).split('e')
    return f"{float(coefficient):.{precision}f} × 10^{int(exponent)}"

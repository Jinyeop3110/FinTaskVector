"""
Program executor for FinQA DSL.

Following the original FinQA implementation:
https://github.com/czyssrs/FinQA

DSL Operations:
- Arithmetic: add, subtract, multiply, divide, exp
- Comparison: greater
- Table: table_sum, table_average, table_max, table_min

Program format: "op1(arg1, arg2), op2(#0, arg3), ..."
- #N refers to the result of step N (0-indexed)
- const_X refers to constant X (e.g., const_100 = 100)
"""

import re
from typing import Optional, Union

# Supported operations
OPERATIONS = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "exp",
    "greater",
    "table_sum",
    "table_average",
    "table_max",
    "table_min",
]

# Common constants used in FinQA
CONSTANTS = {
    "const_1": 1,
    "const_2": 2,
    "const_3": 3,
    "const_4": 4,
    "const_5": 5,
    "const_6": 6,
    "const_7": 7,
    "const_8": 8,
    "const_9": 9,
    "const_10": 10,
    "const_12": 12,
    "const_100": 100,
    "const_1000": 1000,
    "const_1000000": 1000000,
    "const_0.01": 0.01,
    "const_0.1": 0.1,
}


def str_to_num(text: str) -> Optional[float]:
    """
    Convert string to number, following original FinQA implementation.

    Handles:
    - Plain numbers: 123, 123.45, -123
    - Percentages: 12.5%, 12.5 %
    - Formatted: 1,234.56
    - Parentheses negatives: (123)

    Args:
        text: String to convert

    Returns:
        Float value or None if conversion fails
    """
    if not text:
        return None

    text = str(text).strip()

    # Check for constant reference
    if text in CONSTANTS:
        return CONSTANTS[text]

    # Handle "yes"/"no" for greater operation results
    if text.lower() == "yes":
        return 1.0
    if text.lower() == "no":
        return 0.0

    # Remove formatting
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace(" ", "")

    # Handle percentage
    is_percent = False
    if text.endswith("%"):
        is_percent = True
        text = text[:-1]

    # Handle parentheses for negative
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]

    try:
        value = float(text)
        if is_percent:
            value = value / 100
        return value
    except ValueError:
        return None


def process_row(row: list) -> list:
    """
    Process a table row, extracting numeric values.

    Args:
        row: List of cell values

    Returns:
        List of processed values (floats where possible)
    """
    processed = []
    for cell in row:
        num = str_to_num(str(cell))
        if num is not None:
            processed.append(num)
        else:
            processed.append(cell)
    return processed


def tokenize_program(program: str) -> list[str]:
    """
    Tokenize a program string into tokens.

    Args:
        program: Program string like "add(1, 2), subtract(#0, 3)"

    Returns:
        List of tokens
    """
    # Handle empty program
    if not program or program.strip() == "":
        return []

    tokens = []
    current = ""

    for char in program:
        if char in "(),":
            if current.strip():
                tokens.append(current.strip())
            if char != ",":
                tokens.append(char)
            current = ""
        else:
            current += char

    if current.strip():
        tokens.append(current.strip())

    return tokens


def parse_program(program: str) -> list[tuple]:
    """
    Parse a program string into a list of operations.

    Args:
        program: Program string like "add(1, 2), subtract(#0, 3)"

    Returns:
        List of tuples: [(op, arg1, arg2), ...]
    """
    operations = []

    # Split by ), to separate operations
    parts = program.split("),")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Add back the closing paren if needed
        if not part.endswith(")"):
            part += ")"

        # Parse operation: op(arg1, arg2)
        match = re.match(r"(\w+)\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)", part)
        if match:
            op = match.group(1)
            arg1 = match.group(2).strip()
            arg2 = match.group(3).strip()
            operations.append((op, arg1, arg2))

    return operations


def execute_operation(
    op: str,
    arg1: float,
    arg2: float,
) -> Optional[float]:
    """
    Execute a single operation.

    Args:
        op: Operation name
        arg1: First argument
        arg2: Second argument

    Returns:
        Result or None if operation fails
    """
    try:
        if op == "add":
            return arg1 + arg2
        elif op == "subtract":
            return arg1 - arg2
        elif op == "multiply":
            return arg1 * arg2
        elif op == "divide":
            if arg2 == 0:
                return None
            return arg1 / arg2
        elif op == "exp":
            return arg1 ** arg2
        elif op == "greater":
            return 1.0 if arg1 > arg2 else 0.0
        else:
            return None
    except Exception:
        return None


def execute_table_operation(
    op: str,
    table: list[list],
    row_name: str,
) -> Optional[float]:
    """
    Execute a table operation.

    Args:
        op: Table operation name
        table: Table data (list of rows)
        row_name: Name of the row to operate on

    Returns:
        Result or None if operation fails
    """
    # Find the row by name
    values = []
    for row in table:
        if row and str(row[0]).strip().lower() == row_name.strip().lower():
            # Extract numeric values from the row
            for cell in row[1:]:
                num = str_to_num(str(cell))
                if num is not None:
                    values.append(num)
            break

    if not values:
        return None

    try:
        if op == "table_sum":
            return sum(values)
        elif op == "table_average":
            return sum(values) / len(values)
        elif op == "table_max":
            return max(values)
        elif op == "table_min":
            return min(values)
        else:
            return None
    except Exception:
        return None


def execute_program(
    program: str,
    table: Optional[list[list]] = None,
    number_mapping: Optional[dict] = None,
) -> Optional[float]:
    """
    Execute a FinQA program and return the result.

    Following the original FinQA evaluation:
    - Results are rounded to 5 decimal places
    - #N references result of step N
    - const_X references constant X

    Args:
        program: Program string like "add(1, 2), subtract(#0, 3)"
        table: Optional table data for table operations
        number_mapping: Optional mapping of number references to values

    Returns:
        Execution result or None if execution fails
    """
    if not program or program.strip() == "":
        return None

    # Parse the program
    operations = parse_program(program)

    if not operations:
        return None

    # Store results of each step
    results = {}

    for step, (op, arg1_str, arg2_str) in enumerate(operations):
        # Resolve arguments
        def resolve_arg(arg_str: str) -> Optional[float]:
            arg_str = arg_str.strip()

            # Step reference: #0, #1, etc.
            if arg_str.startswith("#"):
                try:
                    ref_idx = int(arg_str[1:])
                    return results.get(ref_idx)
                except ValueError:
                    return None

            # Constant reference: const_100, etc.
            if arg_str.startswith("const_"):
                return CONSTANTS.get(arg_str)

            # Number mapping reference
            if number_mapping and arg_str in number_mapping:
                return number_mapping[arg_str]

            # Direct number
            return str_to_num(arg_str)

        arg1 = resolve_arg(arg1_str)
        arg2 = resolve_arg(arg2_str)

        # Handle table operations
        if op.startswith("table_"):
            # For table operations, arg1 is the row name
            if table is not None:
                result = execute_table_operation(op, table, arg1_str)
            else:
                result = None
        else:
            # Regular operation
            if arg1 is None or arg2 is None:
                return None
            result = execute_operation(op, arg1, arg2)

        if result is None:
            return None

        # Round non-comparison results to 5 decimal places (following original)
        if op != "greater":
            result = round(result, 5)

        results[step] = result

    # Return the last result
    if results:
        return results[max(results.keys())]
    return None


def program_to_answer(
    program: str,
    table: Optional[list[list]] = None,
) -> str:
    """
    Execute program and format the answer.

    Args:
        program: Program string
        table: Optional table data

    Returns:
        Formatted answer string
    """
    result = execute_program(program, table)

    if result is None:
        return ""

    # Handle yes/no for greater operation
    if result == 1.0 and "greater" in program:
        # Check if the last operation is greater
        ops = parse_program(program)
        if ops and ops[-1][0] == "greater":
            return "yes"
    elif result == 0.0 and "greater" in program:
        ops = parse_program(program)
        if ops and ops[-1][0] == "greater":
            return "no"

    # Format number
    if result == int(result):
        return str(int(result))
    else:
        # Remove trailing zeros
        return f"{result:.6f}".rstrip("0").rstrip(".")


def validate_program(program: str) -> tuple[bool, str]:
    """
    Validate a program string.

    Args:
        program: Program string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not program or program.strip() == "":
        return False, "Empty program"

    try:
        operations = parse_program(program)

        if not operations:
            return False, "No valid operations found"

        for op, arg1, arg2 in operations:
            if op not in OPERATIONS:
                return False, f"Unknown operation: {op}"

        return True, ""

    except Exception as e:
        return False, str(e)


def extract_program_from_text(text: str) -> Optional[str]:
    """
    Extract a program from LLM-generated text.

    Looks for patterns like:
    - Program: add(1, 2), subtract(#0, 3)
    - add(1, 2), subtract(#0, 3)

    Args:
        text: LLM-generated text

    Returns:
        Extracted program string or None
    """
    if not text:
        return None

    text = text.strip()

    # Try to find "Program:" prefix
    program_match = re.search(r"[Pp]rogram\s*:\s*(.+?)(?:\n|$)", text)
    if program_match:
        return program_match.group(1).strip()

    # Try to find operation patterns directly
    # Look for something like "op(arg1, arg2)"
    op_pattern = r"(" + "|".join(OPERATIONS) + r")\s*\([^)]+\)"
    matches = re.findall(op_pattern, text)

    if matches:
        # Extract the full program
        full_pattern = r"((?:" + "|".join(OPERATIONS) + r")\s*\([^)]+\)(?:\s*,\s*(?:" + "|".join(OPERATIONS) + r")\s*\([^)]+\))*)"
        full_match = re.search(full_pattern, text)
        if full_match:
            return full_match.group(1).strip()

    return None


if __name__ == "__main__":
    # Test program execution
    test_cases = [
        ("divide(100, 100), divide(3.8, #0)", None, 3.8),
        ("subtract(959.2, 991.1), divide(#0, 991.1)", None, -0.03219),
        ("divide(14001, 26302)", None, 0.53232),
        ("multiply(607, 18.13), multiply(#0, const_1000)", None, 11004910.0),
        ("add(100, 200)", None, 300),
        ("greater(100, 50)", None, 1.0),
    ]

    print("Testing program execution:")
    for program, table, expected in test_cases:
        result = execute_program(program, table)
        status = "✓" if abs(result - expected) < 0.001 else "✗"
        print(f"  {status} {program}")
        print(f"      Expected: {expected}, Got: {result}")

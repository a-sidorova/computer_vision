def sign(val):
    return 0 if val == 0 else (1 if val > 0 else -1)


def is_correct_idx(rows, cols, x, y):
    return 0 <= x < rows and 0 <= y < cols - 1

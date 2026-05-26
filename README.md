# Sudoku-based-chaotic-key-generation


import numpy as np
from PIL import Image
import math
import random

# =========================================================
# Sudoku → Chaotic Key Generator (16x16)
# =========================================================

GRID_SIZE = 16          # 16x16 sudoku
TOTAL_CELLS = GRID_SIZE * GRID_SIZE   # 256
BOX_SIZE = 4            # 4x4 sub-grids

def encode_sudoku(sudoku):
    """Normalize 16x16 sudoku values (1..16) into [0, 1]."""
    arr = np.array(sudoku, dtype=np.float64).flatten()
    assert arr.size == TOTAL_CELLS, f"Expected {TOTAL_CELLS} cells, got {arr.size}"
    return arr / GRID_SIZE     # values in (0, 1]

def init_chaos(encoded):
    """
    Derive (x0, mu) from the encoded sudoku.
    With 256 cells we have more seed material, so we use:
      - x0 from the mean of the encoded values
      - mu from a position-weighted sum, clamped to the chaotic regime
    """
    # x0 in (0, 1), avoid exact 0 or 1 fixed points
    x0 = float(np.sum(encoded) / TOTAL_CELLS)
    x0 = 0.5 if (x0 < 1e-6 or x0 > 1 - 1e-6) else x0

    # Position-weighted sum gives sensitivity to where each value sits
    weights = np.arange(1, TOTAL_CELLS + 1)
    mu_raw = 3.9 + (np.sum(weights * encoded) / (TOTAL_CELLS * TOTAL_CELLS))
    mu = min(mu_raw, 3.999)    # stay in the strongly chaotic regime, away from 4.0

    return x0, mu

def logistic_map(x0, mu, N):
    X = np.zeros(N, dtype=np.float64)
    X[0] = x0
    for i in range(1, N):
        X[i] = mu * X[i - 1] * (1 - X[i - 1])
    return X

def extract_key(X, length):
    """Quantize the chaotic orbit to bytes."""
    return (np.floor(X[:length] * 1e14).astype(np.uint64) % 256).astype(np.uint8)

def generate_sudoku_chaotic_key(sudoku, key_length, warmup=1000):
    encoded = encode_sudoku(sudoku)
    x0, mu = init_chaos(encoded)
    chaos = logistic_map(x0, mu, key_length + warmup)
    chaos = chaos[warmup:]              # discard transient
    return extract_key(chaos, key_length)


# =========================================================
# Image Encryption (RGB)
# =========================================================

def encrypt_image_rgb(image_path, sudoku, output_path):
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    flat_pixels = img_array.flatten()
    key = generate_sudoku_chaotic_key(sudoku, len(flat_pixels))

    encrypted_flat = np.bitwise_xor(flat_pixels, key)
    encrypted_img_array = encrypted_flat.reshape(img_array.shape)

    Image.fromarray(encrypted_img_array.astype(np.uint8)).save(output_path)
    print("Encrypted image saved as:", output_path)

def decrypt_image_rgb(image_path, sudoku, output_path):
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    flat_pixels = img_array.flatten()
    key = generate_sudoku_chaotic_key(sudoku, len(flat_pixels))

    decrypted_flat = np.bitwise_xor(flat_pixels, key)
    decrypted_img_array = decrypted_flat.reshape(img_array.shape)

    Image.fromarray(decrypted_img_array.astype(np.uint8)).save(output_path)
    print("Decrypted image saved as:", output_path)


# =========================================================
# 16x16 Sudoku Generator (backtracking)
# =========================================================

def is_valid_16(grid, row, col, n):
    # row + column check
    for i in range(GRID_SIZE):
        if grid[row][i] == n or grid[i][col] == n:
            return False
    # 4x4 box check
    br, bc = (row // BOX_SIZE) * BOX_SIZE, (col // BOX_SIZE) * BOX_SIZE
    for i in range(BOX_SIZE):
        for j in range(BOX_SIZE):
            if grid[br + i][bc + j] == n:
                return False
    return True

def generate_16x16_sudoku(seed=None):
    """Generate a valid 16x16 sudoku using randomized backtracking."""
    if seed is not None:
        random.seed(seed)
    grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

    def fill(pos=0):
        if pos == TOTAL_CELLS:
            return True
        r, c = divmod(pos, GRID_SIZE)
        nums = list(range(1, GRID_SIZE + 1))
        random.shuffle(nums)
        for n in nums:
            if is_valid_16(grid, r, c, n):
                grid[r][c] = n
                if fill(pos + 1):
                    return True
                grid[r][c] = 0
        return False

    fill()
    return grid


# =========================================================
# Example: a known valid 16x16 sudoku (cyclic-shift construction)
# =========================================================

def make_example_16x16():
    """Build a valid 16x16 sudoku deterministically."""
    grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            grid[r][c] = ((r * BOX_SIZE + r // BOX_SIZE + c) % GRID_SIZE) + 1
    return grid

example_sudoku_16 = make_example_16x16()


# =========================================================
# PSNR
# =========================================================

def calculate_psnr(original_img_path, processed_img_path):
    original  = np.array(Image.open(original_img_path).convert("RGB"), dtype=np.float64)
    processed = np.array(Image.open(processed_img_path).convert("RGB"), dtype=np.float64)
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * math.log10((255.0 * 255.0) / mse)


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    # Verify the example sudoku is valid
    arr = np.array(example_sudoku_16)
    for r in range(GRID_SIZE):
        assert len(set(arr[r])) == GRID_SIZE, f"Row {r} invalid"
        assert len(set(arr[:, r])) == GRID_SIZE, f"Col {r} invalid"
    for br in range(0, GRID_SIZE, BOX_SIZE):
        for bc in range(0, GRID_SIZE, BOX_SIZE):
            box = arr[br:br+BOX_SIZE, bc:bc+BOX_SIZE].flatten()
            assert len(set(box)) == GRID_SIZE, f"Box ({br},{bc}) invalid"
    print("16x16 sudoku validated: rows, columns and 4x4 boxes all unique.")

    # Encrypt / decrypt cycle
    encrypt_image_rgb("logo.png", example_sudoku_16, "encrypted_logo.png")
    decrypt_image_rgb("encrypted_logo.png", example_sudoku_16, "decrypted_logo.png")

    # Sample key
    gen_key = generate_sudoku_chaotic_key(example_sudoku_16, 100)
    print("First 100 key bytes:", gen_key)

    enc_psnr = calculate_psnr("logo.png", "encrypted_logo.png")
    dec_psnr = calculate_psnr("logo.png", "decrypted_logo.png")
    print(f"PSNR original vs encrypted: {enc_psnr:.4f} dB   (lower is better, ~8-10 dB is good)")
    print(f"PSNR original vs decrypted: {dec_psnr}          (should be inf)")
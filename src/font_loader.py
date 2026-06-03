"""
font_loader.py

Parses the font.h C header file to extract the 32 binary character patterns
(5x7 = 35 pixels each) into a NumPy array.
"""
import numpy as np
import re

def load_font(filepath="font.h"):
    '''
    Reads the font.h file and extracts the 32 characters as a numpy array.
    Each character is a 7-element array of bytes, which we convert to a 5x7 binary array,
    and then flatten into a 35-element vector.
    
    Args:
        filepath (str): Path to font.h file
        
    Returns:
        np.ndarray: Array of shape (32, 35) containing binary {0, 1} values.
    '''
    with open(filepath, 'r') as f:
        content = f.read()

    # Find the block inside braces containing the hex values
    match = re.search(r'\{(.*)\}', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find font data block in " + filepath)
        
    block = match.group(1)
    
    # Remove C-style comments (// ...) to avoid capturing hex values in comments
    block = re.sub(r'//.*', '', block)
    
    # Extract all hex values
    hex_values = re.findall(r'0x[0-9A-Fa-f]{2}', block)
    
    if len(hex_values) != 32 * 7:
        raise ValueError(f"Expected {32*7} hex values, got {len(hex_values)}")
        
    # Convert hex strings to integers
    int_values = [int(hx, 16) for hx in hex_values]
    
    # Reshape to (32 chars, 7 rows)
    char_matrix = np.array(int_values).reshape((32, 7))
    
    binary_chars = []
    
    for char_idx in range(32):
        flattened_bits = []
        for row in range(7):
            val = char_matrix[char_idx, row]
            # Get 5 least significant bits
            bits = []
            for bit_pos in range(4, -1, -1):
                bit = (val >> bit_pos) & 1
                bits.append(bit)
            flattened_bits.extend(bits)
        binary_chars.append(flattened_bits)
        
    return np.array(binary_chars, dtype=np.float32)

def print_character(char_array):
    '''
    Prints a single character array as ASCII art.
    
    Args:
        char_array (np.ndarray): Array of shape (35,) with values {0, 1}
    '''
    grid = char_array.reshape((7, 5))
    for row in range(7):
        line = ""
        for col in range(5):
            line += "##" if grid[row, col] >= 0.5 else ".."
        print(line)

def preview_all_fonts(filepath="font.h"):
    '''
    Loads all fonts and prints them to the terminal sequentially for verification.
    '''
    fonts = load_font(filepath)
    print(f"Loaded {fonts.shape[0]} characters of shape {fonts.shape[1]}")
    for i, char in enumerate(fonts):
        print(f"--- Character {i} ---")
        print_character(char)
        print()

if __name__ == "__main__":
    preview_all_fonts("../font.h")

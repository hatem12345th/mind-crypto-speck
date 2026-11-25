"""
SPECK32/64 Lightweight Block Cipher Implementation.

Reference:
- Beaulieu et al. (NSA 2013, 2015): "The SIMON and SPECK Families of Lightweight Block Ciphers"
- Christoph Gohr (CRYPTO 2019): "Improving Attacks on Round-Reduced Speck32/64 using Deep Learning"
- Jimmy Dani, Kalyan Nakka, Nitesh Saxena (arXiv:2405.19683, 2024):
  "Breaking Indistinguishability with Transfer Learning: A First Look at SPECK32/64 Lightweight Block Ciphers"
"""

from typing import Tuple, List, Union
import numpy as np

WORD_SIZE = 16
MASK16 = 0xFFFF
ALPHA = 7
BETA = 2

# Paper Preset Keys (from Appendix A, Table 7)
# K1: 0x59, fd, 06, 41, 5f, 53, db, 99
# K2: 0xfd, fe, 9c, a6, 10, 5c, b9, c7
PAPER_KEY_1_BYTES = bytes([0x59, 0xfd, 0x06, 0x41, 0x5f, 0x53, 0xdb, 0x99])
PAPER_KEY_2_BYTES = bytes([0xfd, 0xfe, 0x9c, 0xa6, 0x10, 0x5c, 0xb9, 0xc7])


def ror16(x: Union[int, np.ndarray], r: int) -> Union[int, np.ndarray]:
    """16-bit right rotation."""
    if isinstance(x, np.ndarray):
        r_np = np.uint16(r)
        return ((x >> r_np) | (x << np.uint16(16 - r))) & np.uint16(MASK16)
    return ((x >> r) | (x << (16 - r))) & MASK16


def rol16(x: Union[int, np.ndarray], r: int) -> Union[int, np.ndarray]:
    """16-bit left rotation."""
    if isinstance(x, np.ndarray):
        r_np = np.uint16(r)
        return ((x << r_np) | (x >> np.uint16(16 - r))) & np.uint16(MASK16)
    return ((x << r) | (x >> (16 - r))) & MASK16


def enc_round(x: Union[int, np.ndarray], y: Union[int, np.ndarray], k: Union[int, np.uint16]) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
    """
    One round of SPECK32 encryption:
    x_{i+1} = ((x_i >> 7) + y_i) ^ k
    y_{i+1} = (y_i << 2) ^ x_{i+1}
    """
    if isinstance(x, np.ndarray):
        x = (ror16(x, ALPHA) + y) & np.uint16(MASK16)
        x = x ^ np.uint16(k)
        y = rol16(y, BETA) ^ x
        return x, y
    
    x = (ror16(x, ALPHA) + y) & MASK16
    x ^= int(k)
    y = rol16(y, BETA) ^ x
    return x, y


def dec_round(x: Union[int, np.ndarray], y: Union[int, np.ndarray], k: Union[int, np.uint16]) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
    """
    Inverse round of SPECK32 decryption:
    y_i = (y_{i+1} ^ x_{i+1}) >> 2
    x_i = ((x_{i+1} ^ k) - y_i) << 7
    """
    if isinstance(x, np.ndarray):
        y_orig = ror16(y ^ x, BETA)
        diff = ((x ^ np.uint16(k)) - y_orig) & np.uint16(MASK16)
        x_orig = rol16(diff, ALPHA)
        return x_orig, y_orig
    
    y_orig = ror16(y ^ x, BETA)
    diff = ((x ^ int(k)) - y_orig) & MASK16
    x_orig = rol16(diff, ALPHA)
    return x_orig, y_orig


def expand_key(master_key: Union[Tuple[int, int, int, int], List[int], bytes, np.ndarray], rounds: int = 22) -> List[int]:
    """
    Key schedule for SPECK32/64.
    master_key: 4 16-bit words (l2, l1, l0, k0) or 8 bytes.
    Returns list of round keys (length = rounds).
    """
    if isinstance(master_key, (bytes, bytearray)):
        # Convert 8 bytes to 4 16-bit words (little-endian uint16)
        words = list(np.frombuffer(master_key, dtype=np.uint16))
    elif isinstance(master_key, np.ndarray):
        words = list(master_key.astype(np.uint16))
    else:
        words = [int(w) & MASK16 for w in master_key]
    
    if len(words) != 4:
        raise ValueError(f"SPECK32/64 requires 4 16-bit key words, got {len(words)}")

    # Gohr convention / NSA spec: words = [k3, k2, k1, k0]
    # k0 is first round key, l = [k2, k1, k0-predecessors]
    # Specifically: ks[0] = k[3], l = [k[2], k[1], k[0]]
    ks = [0] * rounds
    ks[0] = int(words[3])
    l = [int(words[2]), int(words[1]), int(words[0])]

    for i in range(rounds - 1):
        nl, nrk = enc_round(l[i % 3], ks[i], i)
        l[i % 3] = nl
        ks[i + 1] = nrk

    return ks


class Speck32_64:
    """
    SPECK32/64 Block Cipher.
    Supports single-block and vectorized batch encryption/decryption.
    """

    def __init__(self, key: Union[Tuple[int, int, int, int], List[int], bytes, np.ndarray] = None, rounds: int = 22):
        self.rounds = rounds
        if key is not None:
            self.set_key(key, rounds)
        else:
            self.round_keys = []

    def set_key(self, key: Union[Tuple[int, int, int, int], List[int], bytes, np.ndarray], rounds: int = None):
        if rounds is not None:
            self.rounds = rounds
        self.round_keys = expand_key(key, self.rounds)

    def encrypt_block(self, x: int, y: int, rounds: int = None) -> Tuple[int, int]:
        """Encrypt a single 32-bit block (x, y)."""
        r = rounds if rounds is not None else self.rounds
        if r > len(self.round_keys):
            raise ValueError(f"Cipher has only {len(self.round_keys)} round keys, requested {r}")
        for i in range(r):
            x, y = enc_round(x, y, self.round_keys[i])
        return x, y

    def decrypt_block(self, x: int, y: int, rounds: int = None) -> Tuple[int, int]:
        """Decrypt a single 32-bit block (x, y)."""
        r = rounds if rounds is not None else self.rounds
        if r > len(self.round_keys):
            raise ValueError(f"Cipher has only {len(self.round_keys)} round keys, requested {r}")
        for i in reversed(range(r)):
            x, y = dec_round(x, y, self.round_keys[i])
        return x, y

    def encrypt_batch(self, x: np.ndarray, y: np.ndarray, rounds: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized encryption of N blocks.
        x, y: 1D numpy arrays of dtype np.uint16.
        """
        r = rounds if rounds is not None else self.rounds
        cx = x.copy().astype(np.uint16)
        cy = y.copy().astype(np.uint16)
        for i in range(r):
            cx, cy = enc_round(cx, cy, np.uint16(self.round_keys[i]))
        return cx, cy

    def decrypt_batch(self, x: np.ndarray, y: np.ndarray, rounds: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized decryption of N blocks.
        x, y: 1D numpy arrays of dtype np.uint16.
        """
        r = rounds if rounds is not None else self.rounds
        dx = x.copy().astype(np.uint16)
        dy = y.copy().astype(np.uint16)
        for i in reversed(range(r)):
            dx, dy = dec_round(dx, dy, np.uint16(self.round_keys[i]))
        return dx, dy


def check_speck_testvector() -> bool:
    """
    Verify implementation against Beaulieu et al. (NSA 2013) test vector:
    Key:        0x1918 0x1110 0x0908 0x0100
    Plaintext:  0x6574 0x694c
    Ciphertext: 0xa868 0x42f2
    """
    key = (0x1918, 0x1110, 0x0908, 0x0100)
    pt = (0x6574, 0x694c)
    cipher = Speck32_64(key=key, rounds=22)
    ct = cipher.encrypt_block(*pt)
    expected_ct = (0xa868, 0x42f2)

    # Test single block encryption
    if ct != expected_ct:
        return False

    # Test single block decryption
    dec = cipher.decrypt_block(*ct)
    if dec != pt:
        return False

    # Test vectorized encryption & decryption
    x_arr = np.array([pt[0]], dtype=np.uint16)
    y_arr = np.array([pt[1]], dtype=np.uint16)
    vx, vy = cipher.encrypt_batch(x_arr, y_arr)
    if (vx[0], vy[0]) != expected_ct:
        return False
    vdx, vdy = cipher.decrypt_batch(vx, vy)
    if (vdx[0], vdy[0]) != pt:
        return False

    return True


if __name__ == '__main__':
    assert check_speck_testvector(), "SPECK32/64 Testvector failed!"
    print("SPECK32/64 test vector successfully verified!")

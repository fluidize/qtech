import itertools
from mnemonic import Mnemonic
import base58
from bip_utils import Bip39SeedGenerator, Bip44, Bip44Coins
from tqdm import tqdm

# Get the English wordlist
wordlist = Mnemonic("english").wordlist

set2 = [
    "amber",
    "banner",
    "butter",
    "cater",
    "clever",
    "danger",
    "enter",
    "fiber",
    "filter",
    "finger",
    "ladder",
    "later",
    "liver",
    "lumber",
    "master",
    "number",
    "offer",
    "other",
    "paper",
    "powder",
    "proper",
    "recover",
    "remember",
    "river",
    "summer",
    "thunder",
    "under",
    "winter",
    "wonder",
    "worker",
    "zipper",
]

known_words = ["enter", "detect", "pattern", "wing", "primary", "play", "pair", "choose", "shell", "loud"]

# Find all combinations between wordlist and set2
combinations = []
for word1 in wordlist:
    for word2 in set2:
        # Create a combination of word1, word2, and all known_words
        combination = [word1, word2] + known_words
        combinations.append(combination)

def get_solana_address_from_mnemonic(mnemonic):
    # Generate seed from mnemonic
    seed_bytes = Bip39SeedGenerator(mnemonic).Generate()
    
    # Create Solana bip44 object
    bip44_ctx = Bip44.FromSeed(seed_bytes, Bip44Coins.SOLANA)
    
    # Get account 0, address 0
    bip44_account = bip44_ctx.Purpose().Coin().Account(0)
    bip44_change = bip44_account.Change(Bip44.CHANGE_EXT)
    bip44_addr = bip44_change.AddressIndex(0)
    
    # Get public key as base58 string
    pub_key = bip44_addr.PublicKey().RawCompressed().ToBytes()
    return base58.b58encode(pub_key).decode()

# Test with a few combinations
print(f"Testing {len(combinations)} combinations...")
for combination in tqdm(combinations):
    mnemonic = " ".join(combination)
    try:
        address = get_solana_address_from_mnemonic(mnemonic)
        print(f"Found valid mnemonic: {mnemonic}")
        print(f"Address: {address}")
        print("---")
    except Exception:
        # Silent error handling - only print if it works
        pass

const { Keypair } = require('@solana/web3.js');
const bip39 = require('bip39');
const { derivePath } = require('ed25519-hd-key');
const bs58 = require('bs58');
const { SingleBar, Presets } = require('cli-progress');

// English wordlist from BIP39
const wordlist = bip39.wordlists.english;

const set2 = [
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
  "zipper"
];

const knownWords = ["enter", "detect", "pattern", "wing", "primary", "play", "pair", "choose", "shell", "loud"];

// Generate combinations
const combinations = [];
for (const word1 of wordlist) {
  for (const word2 of set2) {
    combinations.push([word1, word2, ...knownWords]);
  }
}

// Function to derive Solana address from mnemonic
function getSolanaAddressFromMnemonic(mnemonic) {
  const seed = bip39.mnemonicToSeedSync(mnemonic);
  
  // Derive using BIP44 path for Solana (m/44'/501'/0'/0')
  const derivedSeed = derivePath("m/44'/501'/0'/0'", seed.toString('hex')).key;
  
  // Create Keypair from the derived seed
  const keypair = Keypair.fromSeed(Uint8Array.from(derivedSeed));
  
  // Return public key as base58 string
  return keypair.publicKey.toBase58();
}

// Progress bar
const bar = new SingleBar({
  format: 'Progress [{bar}] {percentage}% | {value}/{total} combinations',
  barCompleteChar: '\u2588',
  barIncompleteChar: '\u2591',
}, Presets.shades_classic);

console.log(`Testing ${combinations.length} combinations...`);
bar.start(combinations.length, 0);

let count = 0;
for (const combination of combinations) {
  count++;
  bar.update(count);
  
  const mnemonic = combination.join(' ');
  try {
    const address = getSolanaAddressFromMnemonic(mnemonic);
    console.log(`\nFound valid mnemonic: ${mnemonic}`);
    console.log(`Address: ${address}`);
    console.log('---');
  } catch (error) {
    // Silent error handling - only print if it works
  }
}

bar.stop(); 
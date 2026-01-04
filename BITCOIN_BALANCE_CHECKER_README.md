# Bitcoin Balance Checker Scripts

Two scripts for checking Bitcoin wallet balances in **Termux**.

---

## 📱 For Termux Installation

### Prerequisites

```bash
# Update packages
pkg update && pkg upgrade

# Install required packages
pkg install git curl jq bc

# For regtest script, you'll also need Bitcoin Core
# (This is optional and only needed for local regtest network)
```

---

## 🔧 Script 1: Check Regtest Balance (Local)

**File:** `check_regtest_balance.sh`

**Purpose:** Check balance of your LOCAL regtest Bitcoin address

**Usage:**
```bash
./check_regtest_balance.sh
```

**Requirements:**
- bitcoind must be running in regtest mode
- Only works for the regtest address: `bcrt1qv70q00xs5g668wevsh78vq6a75dnh24kgcw99w`

**To start bitcoind in Termux:**
```bash
# This won't work in Termux unless you compile Bitcoin Core for Android
# Regtest is mainly for development on desktop/server systems
```

---

## 🌐 Script 2: Check Real Bitcoin Balance (Mainnet)

**File:** `check_bitcoin_balance.sh`

**Purpose:** Check balance of ANY real Bitcoin address on the blockchain

**Usage:**
```bash
# Check any Bitcoin address
./check_bitcoin_balance.sh <bitcoin-address>

# Examples:
./check_bitcoin_balance.sh 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
./check_bitcoin_balance.sh bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh
./check_bitcoin_balance.sh 3J98t1WpEZ73CNmYviecrnyiWrnqRhWNLy
```

**Features:**
- ✅ Works with Legacy addresses (1...)
- ✅ Works with SegWit addresses (bc1...)
- ✅ Works with P2SH addresses (3...)
- ✅ Shows current balance in BTC
- ✅ Shows total received and sent
- ✅ Shows transaction count
- ✅ Shows last 5 transactions
- ✅ No API key required
- ✅ Works in Termux!

---

## 📲 How to Use in Termux

### Step 1: Copy Scripts to Termux

```bash
# On your Android device in Termux:
cd ~

# Create a bitcoin-tools directory
mkdir -p bitcoin-tools
cd bitcoin-tools

# Download the scripts (if they're in a git repo)
# OR copy them manually using a text editor in Termux
```

### Step 2: Make Scripts Executable

```bash
chmod +x check_bitcoin_balance.sh
chmod +x check_regtest_balance.sh
```

### Step 3: Run the Script

```bash
# For real Bitcoin addresses:
./check_bitcoin_balance.sh YOUR_BITCOIN_ADDRESS

# Example with Satoshi's address:
./check_bitcoin_balance.sh 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
```

---

## 📋 Example Output

```
╔════════════════════════════════════════════════╗
║       Bitcoin Blockchain Balance Checker       ║
╚════════════════════════════════════════════════╝

Checking address: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa

Address Type: Legacy (P2PKH)

Fetching data from blockchain...
═══════════════════════════════════════════════
✓ Balance Information
═══════════════════════════════════════════════

Current Balance:    72.48590000 BTC (7248590000 satoshis)
Total Received:     72.48590000 BTC (7248590000 satoshis)
Total Sent:         0.00000000 BTC (0 satoshis)
Transaction Count:  1534

═══════════════════════════════════════════════

✓ Query completed successfully
```

---

## 🔐 About Your Regtest Address

Your regtest address: `bcrt1qv70q00xs5g668wevsh78vq6a75dnh24kgcw99w`

**Important:** This is a **regtest** address (notice the `bcrt` prefix). It:
- ❌ Does NOT exist on the real Bitcoin blockchain
- ❌ Cannot be checked with blockchain explorers
- ✅ Only exists on your LOCAL regtest network
- ✅ Contains 100.5 BTC (but only in your local test environment)

To check this address, you need to run `check_regtest_balance.sh` on the same machine where bitcoind is running.

---

## 🌍 Alternative: Online Blockchain Explorers

You can also check Bitcoin addresses online without scripts:

**Mainnet:**
- https://blockchain.info/address/YOUR_ADDRESS
- https://blockstream.info/address/YOUR_ADDRESS
- https://mempool.space/address/YOUR_ADDRESS

**Testnet:**
- https://blockstream.info/testnet/address/YOUR_ADDRESS
- https://mempool.space/testnet/address/YOUR_ADDRESS

---

## 🛠️ API Used

The `check_bitcoin_balance.sh` script uses the **blockchain.info** public API:
- API Endpoint: `https://blockchain.info/balance?active=ADDRESS`
- No authentication required
- Rate limited to ~1 request per 10 seconds
- Free for personal use

---

## 📝 Notes

1. **Termux Compatibility:** The mainnet script works perfectly in Termux
2. **No Private Keys:** These scripts only CHECK balances, they cannot spend funds
3. **Read-Only:** Safe to use, no risk to your Bitcoin
4. **Network Required:** Needs internet connection to query blockchain APIs

---

## ⚠️ Troubleshooting

**"command not found: jq"**
```bash
pkg install jq
```

**"command not found: bc"**
```bash
pkg install bc
```

**"command not found: curl"**
```bash
pkg install curl
```

**Script won't run:**
```bash
chmod +x check_bitcoin_balance.sh
./check_bitcoin_balance.sh ADDRESS
```

---

## 🚀 Quick Start (Copy-Paste for Termux)

```bash
# Install dependencies
pkg install curl jq bc -y

# Make script executable
chmod +x check_bitcoin_balance.sh

# Check Satoshi Nakamoto's first address
./check_bitcoin_balance.sh 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
```

---

**Created:** 2025-01-03
**Compatible with:** Termux, Linux, macOS, WSL

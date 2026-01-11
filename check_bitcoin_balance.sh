#!/bin/bash
#
# Bitcoin Balance Checker (Mainnet/Testnet)
# Works with real Bitcoin addresses on the actual blockchain
# Uses blockchain.info API - works in Termux!
#

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if address is provided
if [ -z "$1" ]; then
    echo -e "${RED}Usage: $0 <bitcoin-address>${NC}"
    echo ""
    echo "Example:"
    echo "  $0 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    echo "  $0 bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"
    exit 1
fi

ADDRESS=$1

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Bitcoin Blockchain Balance Checker       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Checking address:${NC} ${CYAN}$ADDRESS${NC}"
echo ""

# Detect address type
if [[ $ADDRESS == bc1* ]]; then
    ADDR_TYPE="SegWit (Bech32)"
elif [[ $ADDRESS == 3* ]]; then
    ADDR_TYPE="P2SH (SegWit compatible)"
elif [[ $ADDRESS == 1* ]]; then
    ADDR_TYPE="Legacy (P2PKH)"
elif [[ $ADDRESS == tb1* ]] || [[ $ADDRESS == bcrt1* ]]; then
    ADDR_TYPE="Testnet/Regtest"
else
    ADDR_TYPE="Unknown"
fi

echo -e "${YELLOW}Address Type:${NC} $ADDR_TYPE"
echo ""

# Query blockchain.info API
echo -e "${BLUE}Fetching data from blockchain...${NC}"

# For testnet/regtest addresses
if [[ $ADDRESS == tb1* ]] || [[ $ADDRESS == bcrt1* ]]; then
    echo -e "${RED}ERROR: This script only works with mainnet addresses${NC}"
    echo "For testnet addresses, use: https://blockstream.info/testnet/api/"
    echo "For regtest addresses, use the check_regtest_balance.sh script"
    exit 1
fi

# Fetch balance data
RESPONSE=$(curl -s "https://blockchain.info/balance?active=$ADDRESS")

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to connect to blockchain.info${NC}"
    exit 1
fi

# Check if address is valid
if echo "$RESPONSE" | grep -q "Invalid Bitcoin Address"; then
    echo -e "${RED}ERROR: Invalid Bitcoin address${NC}"
    exit 1
fi

# Parse JSON response
BALANCE=$(echo "$RESPONSE" | jq -r ".[\"$ADDRESS\"].final_balance // 0")
RECEIVED=$(echo "$RESPONSE" | jq -r ".[\"$ADDRESS\"].total_received // 0")
SENT=$(echo "$RESPONSE" | jq -r ".[\"$ADDRESS\"].total_sent // 0")
TX_COUNT=$(echo "$RESPONSE" | jq -r ".[\"$ADDRESS\"].n_tx // 0")

# Convert satoshis to BTC (divide by 100,000,000)
BALANCE_BTC=$(echo "scale=8; $BALANCE / 100000000" | bc)
RECEIVED_BTC=$(echo "scale=8; $RECEIVED / 100000000" | bc)
SENT_BTC=$(echo "scale=8; $SENT / 100000000" | bc)

# Display results
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Balance Information${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Current Balance:${NC}    ${GREEN}$BALANCE_BTC BTC${NC} (${BALANCE} satoshis)"
echo -e "${YELLOW}Total Received:${NC}     ${CYAN}$RECEIVED_BTC BTC${NC} (${RECEIVED} satoshis)"
echo -e "${YELLOW}Total Sent:${NC}         ${CYAN}$SENT_BTC BTC${NC} (${SENT} satoshis)"
echo -e "${YELLOW}Transaction Count:${NC}  $TX_COUNT"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"

# Get recent transactions (optional)
if [ "$TX_COUNT" -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Fetching recent transactions...${NC}"

    TX_DATA=$(curl -s "https://blockchain.info/rawaddr/$ADDRESS?limit=5")

    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}Recent Transactions (Last 5):${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"

    echo "$TX_DATA" | jq -r '.txs[0:5] | .[] | "  TX: \(.hash)\n  Time: \(.time | strftime("%Y-%m-%d %H:%M:%S"))\n  ───────────────────────────────────────────────"'
fi

echo ""
echo -e "${GREEN}✓ Query completed successfully${NC}"
echo ""

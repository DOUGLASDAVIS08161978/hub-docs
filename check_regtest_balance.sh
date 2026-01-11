#!/bin/bash
#
# Bitcoin Regtest Balance Checker
# This script checks the balance of a regtest Bitcoin address
# Requires bitcoind to be running in regtest mode
#

# Configuration
BITCOIN_CLI="/home/user/hub-docs/bitcoin/build/bin/bitcoin-cli"
DATADIR="/tmp/bitcoin-regtest"
RPC_USER="bitcoinrpc"
RPC_PASSWORD="regtestpassword123"
ADDRESS="bcrt1qv70q00xs5g668wevsh78vq6a75dnh24kgcw99w"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║    Bitcoin Regtest Balance Checker            ║${NC}"
echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo ""

# Check if bitcoind is running
if ! $BITCOIN_CLI -regtest -rpcuser=$RPC_USER -rpcpassword=$RPC_PASSWORD -datadir=$DATADIR getblockchaininfo &> /dev/null; then
    echo -e "${RED}ERROR: bitcoind is not running!${NC}"
    echo ""
    echo "To start bitcoind, run:"
    echo "  $BITCOIN_CLI -regtest -rpcuser=$RPC_USER -rpcpassword=$RPC_PASSWORD -datadir=$DATADIR -daemon"
    exit 1
fi

echo -e "${GREEN}✓ Connected to Bitcoin regtest network${NC}"
echo ""

# Get address info
echo -e "${YELLOW}Address:${NC} $ADDRESS"
echo ""

# Get received amount
RECEIVED=$($BITCOIN_CLI -regtest -rpcuser=$RPC_USER -rpcpassword=$RPC_PASSWORD -datadir=$DATADIR getreceivedbyaddress $ADDRESS 2>/dev/null || echo "0")
echo -e "${YELLOW}Total Received:${NC} ${GREEN}$RECEIVED BTC${NC}"

# List unspent outputs for this address
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Unspent Transaction Outputs (UTXOs):${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"

UNSPENT=$($BITCOIN_CLI -regtest -rpcuser=$RPC_USER -rpcpassword=$RPC_PASSWORD -datadir=$DATADIR listunspent 0 9999999 "[\"$ADDRESS\"]")

if [ "$UNSPENT" = "[]" ]; then
    echo -e "${RED}No unspent outputs found${NC}"
else
    echo "$UNSPENT" | jq -r '.[] | "  TXID: \(.txid)\n  Amount: \(.amount) BTC\n  Confirmations: \(.confirmations)\n  Spendable: \(.spendable)\n  ───────────────────────────────────────────────"'
fi

# Calculate total balance
TOTAL_BALANCE=$($BITCOIN_CLI -regtest -rpcuser=$RPC_USER -rpcpassword=$RPC_PASSWORD -datadir=$DATADIR listunspent 0 9999999 "[\"$ADDRESS\"]" | jq '[.[] | .amount] | add // 0')

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
echo -e "${YELLOW}TOTAL BALANCE:${NC} ${GREEN}$TOTAL_BALANCE BTC${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"

# Get blockchain info
BLOCKCOUNT=$($BITCOIN_CLI -regtest -rpcuser=$RPC_USER -rpcpassword=$RPC_PASSWORD -datadir=$DATADIR getblockcount)
echo ""
echo -e "${YELLOW}Current Block Height:${NC} $BLOCKCOUNT"
echo ""

"""Constants for Bedrock AgentCore Payment SDK."""

from enum import Enum


class PaymentManagerStatus(Enum):
    """Payment manager resource statuses."""

    CREATING = "CREATING"
    UPDATING = "UPDATING"
    DELETING = "DELETING"
    READY = "READY"
    CREATE_FAILED = "CREATE_FAILED"
    UPDATE_FAILED = "UPDATE_FAILED"
    DELETE_FAILED = "DELETE_FAILED"


class PaymentConnectorStatus(Enum):
    """Payment connector statuses."""

    CREATING = "CREATING"
    UPDATING = "UPDATING"
    DELETING = "DELETING"
    READY = "READY"
    CREATE_FAILED = "CREATE_FAILED"
    UPDATE_FAILED = "UPDATE_FAILED"
    DELETE_FAILED = "DELETE_FAILED"
    # Quick Create (QUICK_CREATE provision mode) statuses
    PENDING_AUTHENTICATION = "PENDING_AUTHENTICATION"
    PROVISIONING = "PROVISIONING"
    AUTHENTICATION_EXPIRED = "AUTHENTICATION_EXPIRED"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"


class PaymentConnectorType(Enum):
    """Supported payment connector types."""

    COINBASE_CDP = "CoinbaseCDP"
    STRIPE_PRIVY = "StripePrivy"


class PaymentConnectorProvisionMode(Enum):
    """Payment connector provisioning modes.

    MANUAL (the default) requires the caller to supply the credential provider
    configuration up front. QUICK_CREATE lets the service orchestrate OAuth
    consent and provision the credential provider on the caller's behalf.
    """

    MANUAL = "MANUAL"
    QUICK_CREATE = "QUICK_CREATE"


class PaymentType(Enum):
    """Payment protocols supported by ProcessPayment."""

    CRYPTO_X402 = "CRYPTO_X402"
    # MPP does not differentiate between crypto and fiat — one value covers both.
    MPP = "MPP"


class PaymentsAuthorizerType(Enum):
    """Payment manager authorizer types."""

    CUSTOM_JWT = "CUSTOM_JWT"
    AWS_IAM = "AWS_IAM"


# Default constants
DEFAULT_MAX_RESULTS = 100

# Define network preference order (most preferred first)
NETWORK_PREFERENCES = [
    # Solan first as it is fast and low cost
    "solana-mainnet",  # Solana Mainnet (simplified identifier)
    "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp",  # Mainnet genesis hash (32 chars, CAIP-2)
    "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d",  # Mainnet full genesis hash (44 chars)
    # Ethereum network
    "eip155:8453",  # Base mainnet (low fees)
    "eip155:1",  # Ethereum mainnet
    "base",
    "eip155:42161",  # Arbitrum One
    "eip155:10",  # Optimism
    "ethereum",
    # SOLANA test network
    "solana-devnet",  # Solana Devnet (simplified identifier)
    "solana:EtWTRABZaYq6iMfeYKouRu166VU2xqa1",  # Devnet genesis hash (32 chars, CAIP-2)
    "solana:EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG",  # Devnet full genesis hash (44 chars)
    "solana-testnet",  # Solana Testnet (simplified identifier)
    "solana:4uhcVJyU9pJkvQyS88uRDiswHXSCkY3z",  # Testnet genesis hash (32 chars, CAIP-2)
    "solana:4uhcVJyU9pJkvQyS88uRDiswHXSCkY3zQawwpjk2NsNY",  # Testnet full genesis hash (44 chars)
    # Ethereum test
    "sepolia",
    "base-sepolia",
    "eip155:84532",  # Base Sepolia (testnet)
    "eip155:11155111",  # Ethereum Sepolia (Test)
]

# ─────────────────────────────────────────────────────────────
# MPP (Machine Payments Protocol) — https://mpp.dev
# ─────────────────────────────────────────────────────────────

# Default MPP protocol version sent to ProcessPayment. The service model constrains
# this to a bare numeric string (^[0-9]+$).
MPP_DEFAULT_VERSION = "1"

# The `Payment` auth-scheme name used in `WWW-Authenticate` / `Authorization` headers.
MPP_AUTH_SCHEME = "Payment"

# Only the `charge` intent is implemented. Challenges advertising `session` or
# `subscription` are filtered out during selection.
MPP_SUPPORTED_INTENT = "charge"

# Payment method identifiers, mapped to the blockchain family of the payment
# instrument that can satisfy them. Tempo is an EVM chain, so it maps to ETHEREUM.
MPP_METHOD_BLOCKCHAIN = {
    "evm": "ETHEREUM",
    "tempo": "ETHEREUM",
    "solana": "SOLANA",
}

# Maps a Solana `methodDetails.network` value to the network identifier used in
# NETWORK_PREFERENCES. Per draft-solana-charge-00 the field is optional and
# defaults to mainnet.
#
# `localnet` is deliberately absent. It denotes a distinct local RPC/Surfpool
# environment, not Solana testnet; aliasing it would let a local-only challenge
# satisfy a solana-testnet preference and outrank a genuinely payable devnet
# challenge. Unmapped values are left unranked rather than misranked — they remain
# selectable when they are the only option, but never win on preference order.
MPP_SOLANA_NETWORK_ALIASES = {
    "mainnet": "solana-mainnet",
    "devnet": "solana-devnet",
    "testnet": "solana-testnet",
}
MPP_SOLANA_DEFAULT_NETWORK = "mainnet"

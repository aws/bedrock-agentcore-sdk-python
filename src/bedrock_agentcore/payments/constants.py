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


class PaymentConnectorType(Enum):
    """Supported payment connector types."""

    COINBASE_CDP = "CoinbaseCDP"
    STRIPE_PRIVY = "StripePrivy"


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

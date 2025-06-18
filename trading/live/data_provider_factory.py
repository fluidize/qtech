from typing import Dict, Any
from trading.live.providers import BaseDataProvider
from providers import BinanceProvider
from providers import KuCoinProvider

class DataProviderFactory:
    """Factory class to create data providers."""
    
    PROVIDERS = {
        'binance': BinanceProvider,
        'kucoin': KuCoinProvider,
    }
    
    @classmethod
    def create_provider(cls, data_source: str, symbol: str, **kwargs) -> BaseDataProvider:
        """
        Create a data provider instance.
        
        Args:
            data_source: Provider name ('binance', 'kucoin')
            symbol: Trading symbol
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Data provider instance
            
        Raises:
            ValueError: If data source is not supported
        """
        if data_source.lower() not in cls.PROVIDERS:
            available = ', '.join(cls.PROVIDERS.keys())
            raise ValueError(f"Unsupported data source: {data_source}. Available: {available}")
        
        provider_class = cls.PROVIDERS[data_source.lower()]
        return provider_class(symbol, **kwargs)
    
    @classmethod
    def get_supported_providers(cls) -> list:
        """Get list of supported data providers."""
        return list(cls.PROVIDERS.keys())
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """
        Register a new data provider.
        
        Args:
            name: Provider name
            provider_class: Provider class (must inherit from BaseDataProvider)
        """
        if not issubclass(provider_class, BaseDataProvider):
            raise ValueError("Provider class must inherit from BaseDataProvider")
        
        cls.PROVIDERS[name.lower()] = provider_class 
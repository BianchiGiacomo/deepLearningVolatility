# -*- coding: utf-8 -*-
"""
Pricing backend interface for external pricing libraries integration.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class PricingBackend(ABC):
    """
    Abstract interface for external pricing backends.
    Allows integration with proprietary pricing libraries.
    """
    
    @abstractmethod
    def price(self, 
              theta: Dict[str, float], 
              T: float, 
              K: float, 
              callput: str,
              forward: float, 
              disc: float) -> float:
        """
        Calculate option price using external backend.
        
        Args:
            theta: Model parameters as dictionary (e.g., {'kappa': 1.0, 'theta': 0.04, ...})
            T: Time to maturity
            K: Strike price
            callput: 'C' for call, 'P' for put
            forward: Forward price
            disc: Discount factor
            
        Returns:
            Option price
        """
        pass
    
    @abstractmethod
    def implied_vol(self, 
                    T: float, 
                    F: float, 
                    K: float, 
                    price: float,
                    callput: str, 
                    disc: float) -> float:
        """
        Calculate implied volatility from option price.
        
        Args:
            T: Time to maturity
            F: Forward price
            K: Strike price
            price: Option price
            callput: 'C' for call, 'P' for put
            disc: Discount factor
            
        Returns:
            Implied volatility
        """
        pass


class MockPricingBackend(PricingBackend):
    """
    Mock implementation for testing purposes.
    Returns simplified Black-Scholes approximations.
    """
    
    def __init__(self, default_vol: float = 0.2):
        self.default_vol = default_vol
    
    def price(self, 
              theta: Dict[str, float], 
              T: float, 
              K: float, 
              callput: str,
              forward: float, 
              disc: float) -> float:
        """
        Mock pricing - returns a simplified BS price.
        """
        import numpy as np
        from scipy.stats import norm
        
        # Use theta's volatility parameter if available
        vol = self.default_vol
        for key in ['sigma', 'xi0', 'theta', 'vol']:
            if key in theta:
                vol = float(theta[key])
                if key in ['xi0', 'theta']:  # These are variance parameters
                    vol = np.sqrt(vol)
                break
        
        # Simplified Black-Scholes
        log_moneyness = np.log(forward / K)
        d1 = (log_moneyness + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        
        if callput == 'C':
            price = forward * norm.cdf(d1) - K * norm.cdf(d2)
        else:
            price = K * norm.cdf(-d2) - forward * norm.cdf(-d1)
        
        return price * disc
    
    def implied_vol(self, 
                    T: float, 
                    F: float, 
                    K: float, 
                    price: float,
                    callput: str, 
                    disc: float) -> float:
        """
        Mock IV calculation - returns default volatility.
        In production, this would use Newton-Raphson or similar.
        """
        return self.default_vol
"""
ETF and Mutual Funds Integration
Loads ETF/MF data and provides investment recommendations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ETFMFIntegration:
    """Integration for ETF and Mutual Funds recommendations"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.etf_data = None
        self.mf_data = None
        self.load_data()
    
    def load_data(self):
        """Load ETF and Mutual Funds data"""
        try:
            # Load ETF data - try multiple possible paths
            etf_paths = [
                self.data_dir / "data/ETF and MF data/ETFs.csv",
                self.data_dir / "data/Us funds/MF and ETF data/ETFs.csv",
                self.data_dir / "data/ETF and MF data/ETF prices.csv"
            ]
            
            etf_data_loaded = False
            for etf_path in etf_paths:
                if etf_path.exists():
                    self.etf_data = pd.read_csv(etf_path)
                    print(f"Loaded {len(self.etf_data)} ETFs from {etf_path.name}")
                    etf_data_loaded = True
                    break
            
            if not etf_data_loaded:
                print("ETF data file not found - using mock data")
                self._create_mock_etf_data()
            
            # Load Mutual Fund data if available
            mf_paths = [
                self.data_dir / "data/Us funds/MF and ETF data",
                self.data_dir / "data/ETF and MF data"
            ]
            
            for mf_path in mf_paths:
                if mf_path.exists():
                    mf_files = list(mf_path.glob("*.csv"))
                    if mf_files:
                        self.mf_data = pd.concat([pd.read_csv(f) for f in mf_files[:3]])  # Limit to 3 files
                        print(f"Loaded {len(self.mf_data)} mutual fund records")
                        break
            
            if self.mf_data is None:
                print("Mutual fund data not found - using mock data")
                self._create_mock_mf_data()
            
        except Exception as e:
            print(f"Error loading ETF/MF data: {e}")
            # Create mock data as fallback
            self._create_mock_etf_data()
            self._create_mock_mf_data()
    
    def _create_mock_etf_data(self):
        """Create mock ETF data when real data is not available"""
        mock_etfs = [
            {'Name': 'SPDR S&P 500 ETF', 'Symbol': 'SPY', 'Category': 'Large Cap', 'Expense Ratio': 0.09},
            {'Name': 'Invesco QQQ Trust', 'Symbol': 'QQQ', 'Category': 'Technology', 'Expense Ratio': 0.18},
            {'Name': 'iShares Core MSCI EAFE', 'Symbol': 'IEFA', 'Category': 'International', 'Expense Ratio': 0.07},
            {'Name': 'Vanguard Total Bond Market', 'Symbol': 'BND', 'Category': 'Bonds', 'Expense Ratio': 0.03},
            {'Name': 'iShares Gold Trust', 'Symbol': 'GLD', 'Category': 'Commodities', 'Expense Ratio': 0.40},
            {'Name': 'Vanguard Real Estate ETF', 'Symbol': 'VNQ', 'Category': 'Real Estate', 'Expense Ratio': 0.12},
            {'Name': 'iShares MSCI Emerging Markets', 'Symbol': 'EEM', 'Category': 'Emerging Markets', 'Expense Ratio': 0.69},
            {'Name': 'Vanguard Dividend Appreciation', 'Symbol': 'VIG', 'Category': 'Dividend', 'Expense Ratio': 0.06},
            {'Name': 'iShares Core S&P Mid-Cap', 'Symbol': 'IJH', 'Category': 'Mid Cap', 'Expense Ratio': 0.05},
            {'Name': 'Vanguard Growth ETF', 'Symbol': 'VUG', 'Category': 'Growth', 'Expense Ratio': 0.04}
        ]
        
        self.etf_data = pd.DataFrame(mock_etfs)
        print(f"Created {len(self.etf_data)} mock ETF records")
    
    def _create_mock_mf_data(self):
        """Create mock Mutual Fund data when real data is not available"""
        mock_mfs = [
            {'Name': 'Vanguard 500 Index', 'Category': 'Large Cap', 'Expense Ratio': 0.14, 'Min Investment': 3000},
            {'Name': 'Fidelity Contrafund', 'Category': 'Growth', 'Expense Ratio': 0.85, 'Min Investment': 2500},
            {'Name': 'T. Rowe Price Equity Income', 'Category': 'Value', 'Expense Ratio': 0.69, 'Min Investment': 2500},
            {'Name': 'American Funds Growth Fund', 'Category': 'Growth', 'Expense Ratio': 0.64, 'Min Investment': 250},
            {'Name': 'Vanguard Total Bond Market', 'Category': 'Bonds', 'Expense Ratio': 0.11, 'Min Investment': 3000}
        ]
        
        self.mf_data = pd.DataFrame(mock_mfs)
        print(f"Created {len(self.mf_data)} mock mutual fund records")
    
    def get_investment_recommendations(self, user_profile):
        """Get investment recommendations based on user profile"""
        recommendations = []
        
        if self.etf_data is None and self.mf_data is None:
            return self._get_generic_recommendations(user_profile)
        
        income = user_profile.get('total_income', 0)
        cluster = user_profile.get('consensus_cluster_name', 'Middle Income Families')
        savings_rate = user_profile.get('savings_rate', 0)
        
        # Risk tolerance based on cluster
        risk_tolerance = self._get_risk_tolerance(cluster)
        
        # ETF recommendations
        if self.etf_data is not None:
            etf_recs = self._get_etf_recommendations(income, risk_tolerance, savings_rate)
            recommendations.extend(etf_recs)
        
        # Mutual Fund recommendations
        if self.mf_data is not None:
            mf_recs = self._get_mf_recommendations(income, risk_tolerance, cluster)
            recommendations.extend(mf_recs)
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _get_risk_tolerance(self, cluster):
        """Determine risk tolerance from cluster"""
        risk_map = {
            'High Income Savers': 'High',
            'Middle Income Families': 'Medium', 
            'Zero Income Households': 'Low'
        }
        return risk_map.get(cluster, 'Medium')
    
    def _get_etf_recommendations(self, income, risk_tolerance, savings_rate):
        """Get ETF recommendations"""
        if self.etf_data is None:
            return []
        
        # Filter ETFs based on criteria
        suitable_etfs = self.etf_data.copy()
        
        # Filter by risk level (simplified)
        if risk_tolerance == 'Low':
            suitable_etfs = suitable_etfs.head(10)  # Conservative ETFs
        elif risk_tolerance == 'Medium':
            suitable_etfs = suitable_etfs.head(20)  # Balanced ETFs
        else:
            suitable_etfs = suitable_etfs.head(30)  # Growth ETFs
        
        # Create recommendations
        recommendations = []
        for _, etf in suitable_etfs.head(3).iterrows():
            recommendations.append({
                'type': 'ETF',
                'name': etf.get('Name', 'Unknown ETF'),
                'symbol': etf.get('Symbol', 'N/A'),
                'category': etf.get('Category', 'Diversified'),
                'risk_level': risk_tolerance,
                'reason': f'Suitable for {risk_tolerance.lower()} risk tolerance',
                'min_investment': max(1000, income * 0.01)  # 1% of income
            })
        
        return recommendations
    
    def _get_mf_recommendations(self, income, risk_tolerance, cluster):
        """Get Mutual Fund recommendations"""
        if self.mf_data is None:
            return []
        
        # Simple mutual fund recommendations
        recommendations = []
        
        # Based on cluster and income
        if cluster == 'High Income Savers':
            recommendations.append({
                'type': 'Mutual Fund',
                'name': 'Growth Equity Fund',
                'category': 'Equity',
                'risk_level': 'High',
                'reason': 'Growth potential for high income',
                'min_investment': max(5000, income * 0.05)
            })
        elif cluster == 'Middle Income Families':
            recommendations.append({
                'type': 'Mutual Fund', 
                'name': 'Balanced Fund',
                'category': 'Hybrid',
                'risk_level': 'Medium',
                'reason': 'Balanced growth and stability',
                'min_investment': max(2500, income * 0.03)
            })
        else:
            recommendations.append({
                'type': 'Mutual Fund',
                'name': 'Conservative Bond Fund', 
                'category': 'Debt',
                'risk_level': 'Low',
                'reason': 'Capital preservation focus',
                'min_investment': max(1000, income * 0.02)
            })
        
        return recommendations
    
    def _get_generic_recommendations(self, user_profile):
        """Fallback generic recommendations"""
        income = user_profile.get('total_income', 0)
        cluster = user_profile.get('consensus_cluster_name', 'Middle Income Families')
        
        return [
            {
                'type': 'Generic Investment',
                'name': 'Diversified Portfolio',
                'category': 'Mixed',
                'risk_level': self._get_risk_tolerance(cluster),
                'reason': 'Start with diversified investments',
                'min_investment': max(1000, income * 0.02)
            }
        ]

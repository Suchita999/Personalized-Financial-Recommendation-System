"""
K-means Clustering for CE Interview Data
Customer segmentation for financial product recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class KMeansClusterer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.features_scaled = None
        self.scaler = None
        self.kmeans = None
        self.cluster_labels = None
        self.optimal_k = None
        self.cluster_stats = None
        
    def load_data(self):
        """Load the fixed engineered features"""
        print("Loading engineered features...")
        self.data = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.data):,} records with {self.data.shape[1]} features")
        
    def prepare_features(self, exclude_cols=None):
        """Prepare features for clustering"""
        print("Preparing features for clustering...")
        
        if exclude_cols is None:
            exclude_cols = ['NEWID', 'quarter', 'financial_health_tier', 
                          'primary_spending_category', 'age_group', 'region',
                          'housing_tenure', 'education_level', 'marital_status', 'race']
        
        # Select numeric features only
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        print(f"✓ Selected {len(feature_cols)} features for clustering")
        
        # Handle any remaining missing values
        features = self.data[feature_cols].copy()
        features = features.fillna(features.median())
        
        # Standardize features
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(features)
        
        print(f"✓ Features standardized (mean=0, std=1)")
        return feature_cols
    
    def find_optimal_k(self, max_k=10, plot=True):
        """Find optimal number of clusters using elbow method and silhouette score"""
        print("Finding optimal number of clusters...")
        
        k_range = range(2, max_k + 1)
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.features_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.features_scaled, labels))
            calinski_scores.append(calinski_harabasz_score(self.features_scaled, labels))
            
            print(f"  K={k}: Inertia={kmeans.inertia_:.0f}, "
                  f"Silhouette={silhouette_score(self.features_scaled, labels):.3f}")
        
        # Find optimal k based on silhouette score
        self.optimal_k = k_range[np.argmax(silhouette_scores)]
        
        if plot:
            self._plot_optimal_k_analysis(k_range, inertias, silhouette_scores, calinski_scores)
        
        print(f"✓ Optimal K = {self.optimal_k} (highest silhouette score)")
        return self.optimal_k
    
    def _plot_optimal_k_analysis(self, k_range, inertias, silhouette_scores, calinski_scores):
        """Plot analysis for optimal K selection"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Elbow plot
        axes[0, 0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Number of Clusters (K)')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Method')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Silhouette score
        axes[0, 1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Number of Clusters (K)')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Score Analysis')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Calinski-Harabasz score
        axes[1, 0].plot(k_range, calinski_scores, 'mo-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Number of Clusters (K)')
        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
        axes[1, 0].set_title('Calinski-Harabasz Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined view
        axes[1, 1].plot(k_range, silhouette_scores, 'go-', label='Silhouette', linewidth=2)
        axes[1, 1].axvline(x=self.optimal_k, color='red', linestyle='--', 
                           label=f'Optimal K={self.optimal_k}')
        axes[1, 1].set_xlabel('Number of Clusters (K)')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Optimal K Selection')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./feature-engineering-output/optimal_k_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def fit_kmeans(self, k=None, random_state=42):
        """Fit K-means with specified or optimal K"""
        if k is None:
            k = self.optimal_k
        
        print(f"Fitting K-means with K={k}...")
        
        self.kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.features_scaled)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(self.features_scaled, self.cluster_labels)
        calinski_avg = calinski_harabasz_score(self.features_scaled, self.cluster_labels)
        
        print(f"✓ K-means fitted successfully")
        print(f"  Silhouette Score: {silhouette_avg:.3f}")
        print(f"  Calinski-Harabasz Score: {calinski_avg:.0f}")
        print(f"  Inertia: {self.kmeans.inertia_:.0f}")
        
        return self.cluster_labels
    
    def analyze_clusters(self, feature_cols):
        """Analyze cluster characteristics"""
        print("Analyzing cluster characteristics...")
        
        # Add cluster labels to original data
        data_with_clusters = self.data.copy()
        data_with_clusters['cluster'] = self.cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = {}
        
        for cluster_id in range(self.kmeans.n_clusters):
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
            
            stats = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data_with_clusters) * 100,
            }
            
            # Key financial metrics
            key_metrics = ['total_income', 'total_expenditure', 'savings_rate', 
                         'expenditure_to_income_ratio', 'age_ref', 'family_size']
            
            for metric in key_metrics:
                if metric in cluster_data.columns:
                    stats[f'{metric}_mean'] = cluster_data[metric].mean()
                    stats[f'{metric}_median'] = cluster_data[metric].median()
            
            # Binary features percentages
            binary_features = ['is_homeowner', 'is_married', 'is_positive_savings', 
                            'high_spender', 'high_income', 'zero_income_flag']
            
            for feature in binary_features:
                if feature in cluster_data.columns:
                    stats[f'{feature}_pct'] = cluster_data[feature].mean() * 100
            
            cluster_stats[cluster_id] = stats
        
        self.cluster_stats = pd.DataFrame(cluster_stats).T
        
        print("✓ Cluster analysis complete")
        return self.cluster_stats
    
    def visualize_clusters(self):
        """Visualize clusters using PCA"""
        print("Creating cluster visualizations...")
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(self.features_scaled)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Scatter plot with clusters
        scatter = axes[0, 0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                    c=self.cluster_labels, cmap='viridis', alpha=0.6, s=50)
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[0, 0].set_title('K-means Clusters (PCA Visualization)')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # Cluster centers
        centers_2d = pca.transform(self.kmeans.cluster_centers_)
        axes[0, 1].scatter(features_2d[:, 0], features_2d[:, 1], 
                          c=self.cluster_labels, cmap='viridis', alpha=0.3, s=30)
        axes[0, 1].scatter(centers_2d[:, 0], centers_2d[:, 1], 
                          c='red', marker='X', s=200, linewidths=2, edgecolors='black')
        axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[0, 1].set_title('Cluster Centers')
        
        # Cluster sizes
        cluster_sizes = pd.Series(self.cluster_labels).value_counts().sort_index()
        colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_sizes)))
        axes[1, 0].bar(cluster_sizes.index, cluster_sizes.values, color=colors)
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Number of Households')
        axes[1, 0].set_title('Cluster Sizes')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Income vs Expenditure by cluster
        data_with_clusters = self.data.copy()
        data_with_clusters['cluster'] = self.cluster_labels
        
        for cluster_id in range(self.kmeans.n_clusters):
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
            axes[1, 1].scatter(cluster_data['total_income'], cluster_data['total_expenditure'], 
                              label=f'Cluster {cluster_id}', alpha=0.6, s=30)
        
        axes[1, 1].set_xlabel('Total Income')
        axes[1, 1].set_ylabel('Total Expenditure')
        axes[1, 1].set_title('Income vs Expenditure by Cluster')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./feature-engineering-output/kmeans_visualizations.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_cluster_profiles(self):
        """Create detailed cluster profiles for business interpretation"""
        print("Creating cluster profiles...")
        
        profiles = []
        
        for cluster_id in range(self.kmeans.n_clusters):
            stats = self.cluster_stats.loc[cluster_id]
            
            # Determine cluster characteristics
            if stats['zero_income_flag_pct'] > 50:
                profile_type = "Zero Income Households"
                key_features = f"No income, ${stats['total_expenditure_median']:,.0f} avg expenditure"
            elif stats['high_income_pct'] > 50:
                if stats['savings_rate_mean'] > 0.2:
                    profile_type = "High Income Savers"
                    key_features = f"${stats['total_income_median']:,.0f} income, {stats['savings_rate_mean']:.1%} savings"
                else:
                    profile_type = "High Income Spenders"
                    key_features = f"${stats['total_income_median']:,.0f} income, high spending"
            elif stats['is_positive_savings_pct'] < 30:
                profile_type = "Financially Stretched"
                key_features = f"${stats['total_income_median']:,.0f} income, negative savings"
            elif stats['age_ref_mean'] < 35:
                profile_type = "Young Adults"
                key_features = f"Age {stats['age_ref_mean']:.0f}, ${stats['total_income_median']:,.0f} income"
            elif stats['is_homeowner_pct'] > 70:
                profile_type = "Established Homeowners"
                key_features = f"{stats['is_homeowner_pct']:.0f}% homeowners, stable finances"
            else:
                profile_type = "Middle Income Families"
                key_features = f"${stats['total_income_median']:,.0f} income, balanced spending"
            
            profiles.append({
                'Cluster': cluster_id,
                'Profile_Type': profile_type,
                'Size': f"{stats['size']:,} ({stats['percentage']:.1f}%)",
                'Key_Characteristics': key_features,
                'Avg_Income': f"${stats['total_income_mean']:,.0f}",
                'Avg_Expenditure': f"${stats['total_expenditure_mean']:,.0f}",
                'Savings_Rate': f"{stats['savings_rate_mean']:.1%}",
                'Homeowners': f"{stats['is_homeowner_pct']:.0f}%",
                'Positive_Savings': f"{stats['is_positive_savings_pct']:.0f}%"
            })
        
        profiles_df = pd.DataFrame(profiles)
        
        print("✓ Cluster profiles created")
        return profiles_df
    
    def save_results(self, output_dir='./feature-engineering-output'):
        """Save all clustering results"""
        print("Saving clustering results...")
        
        # Save data with cluster labels
        data_with_clusters = self.data.copy()
        data_with_clusters['cluster'] = self.cluster_labels
        data_with_clusters.to_csv(f'{output_dir}/clustered_households.csv', index=False)
        
        # Save cluster statistics
        self.cluster_stats.to_csv(f'{output_dir}/cluster_statistics.csv')
        
        # Save cluster profiles
        profiles_df = self.create_cluster_profiles()
        profiles_df.to_csv(f'{output_dir}/cluster_profiles.csv', index=False)
        
        print(f"✓ Results saved to {output_dir}")
        return profiles_df
    
    def run_complete_clustering(self, max_k=10):
        """Run complete clustering pipeline"""
        print("Starting K-means clustering pipeline...")
        
        # Load and prepare data
        self.load_data()
        feature_cols = self.prepare_features()
        
        # Find optimal K
        self.find_optimal_k(max_k=max_k)
        
        # Fit K-means
        self.fit_kmeans()
        
        # Analyze clusters
        self.analyze_clusters(feature_cols)
        
        # Create visualizations
        self.visualize_clusters()
        
        # Save results
        profiles_df = self.save_results()
        
        print("✓ K-means clustering complete!")
        return profiles_df

def main():
    """Main function to run K-means clustering"""
    # Initialize clusterer
    data_path = './feature-engineering-output/engineered_features_fixed.csv'
    clusterer = KMeansClusterer(data_path)
    
    # Run complete clustering
    profiles_df = clusterer.run_complete_clustering(max_k=10)
    
    # Display cluster profiles
    print("\n" + "="*80)
    print("CLUSTER PROFILES FOR FINANCIAL PRODUCT RECOMMENDATIONS")
    print("="*80)
    print(profiles_df.to_string(index=False))
    
    return clusterer, profiles_df

if __name__ == "__main__":
    clusterer, profiles_df = main()

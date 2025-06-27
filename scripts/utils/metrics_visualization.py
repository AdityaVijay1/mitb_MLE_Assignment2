import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import psycopg2
from psycopg2 import sql
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MetricsVisualizer:
    def __init__(self, host='localhost', port=5432, database='airflow', 
                 user='airflow', password='airflow'):
        """
        Initialize the metrics visualizer with PostgreSQL connection.
        """
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        
    def connect(self):
        """Establish connection to PostgreSQL."""
        try:
            conn = psycopg2.connect(**self.connection_params)
            return conn
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
            return None
    
    def get_airflow_dag_metrics(self, days_back=30):
        """
        Get Airflow DAG execution metrics from the database.
        """
        conn = self.connect()
        if not conn:
            return None
            
        query = """
        SELECT 
            dag_id,
            execution_date,
            state,
            start_date,
            end_date,
            EXTRACT(EPOCH FROM (end_date - start_date)) as duration_seconds
        FROM dag_run 
        WHERE execution_date >= NOW() - INTERVAL '%s days'
        ORDER BY execution_date
        """
        
        try:
            df = pd.read_sql_query(query, conn, params=[days_back])
            conn.close()
            return df
        except Exception as e:
            print(f"Error querying Airflow metrics: {e}")
            conn.close()
            return None
    
    def get_model_performance_metrics(self, days_back=30):
        """
        Get model performance metrics from the model_metrics table.
        """
        conn = self.connect()
        if not conn:
            return None
            
        query = """
        SELECT 
            month_date,
            model_name,
            accuracy,
            precision,
            recall,
            macro_f1,
            auc,
            total_samples,
            created_at
        FROM model_metrics 
        WHERE created_at >= NOW() - INTERVAL '%s days'
        ORDER BY month_date
        """
        
        try:
            df = pd.read_sql_query(query, conn, params=[days_back])
            conn.close()
            return df
        except Exception as e:
            print(f"Error querying model metrics: {e}")
            conn.close()
            return None
    
    def get_model_inference_metrics(self, days_back=30):
        """
        Get model inference metrics from the model_inference table.
        """
        conn = self.connect()
        if not conn:
            return None
            
        query = """
        SELECT 
            month_date,
            model_name,
            model_version,
            total_predictions,
            positive_predictions,
            negative_predictions,
            created_at
        FROM model_inference 
        WHERE created_at >= NOW() - INTERVAL '%s days'
        ORDER BY month_date
        """
        
        try:
            df = pd.read_sql_query(query, conn, params=[days_back])
            conn.close()
            return df
        except Exception as e:
            print(f"Error querying inference metrics: {e}")
            conn.close()
            return None
    
    def plot_dag_execution_timeline(self, days_back=30):
        """
        Plot DAG execution timeline showing success/failure over time.
        """
        df = self.get_airflow_dag_metrics(days_back)
        if df is None or df.empty:
            print("No DAG execution data found.")
            return
        
        # Convert execution_date to datetime if it's not already
        df['execution_date'] = pd.to_datetime(df['execution_date'])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: DAG Status Timeline
        for state in df['state'].unique():
            state_data = df[df['state'] == state]
            color = 'green' if state == 'success' else 'red' if state == 'failed' else 'orange'
            ax1.scatter(state_data['execution_date'], state_data['dag_id'], 
                       c=color, label=state, alpha=0.7, s=50)
        
        ax1.set_title('DAG Execution Status Timeline', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Execution Date')
        ax1.set_ylabel('DAG ID')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Execution Duration
        successful_runs = df[df['state'] == 'success'].copy()
        if not successful_runs.empty:
            ax2.plot(successful_runs['execution_date'], successful_runs['duration_seconds'], 
                    marker='o', linewidth=2, markersize=4)
            ax2.set_title('DAG Execution Duration (Successful Runs)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Execution Date')
            ax2.set_ylabel('Duration (seconds)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_performance_trends(self, days_back=30):
        """
        Plot model performance metrics over time.
        """
        df = self.get_model_performance_metrics(days_back)
        if df is None or df.empty:
            print("No model performance data found.")
            return
        
        # Convert month_date to datetime
        df['month_date'] = pd.to_datetime(df['month_date'])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Metrics Over Time', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy and F1 Score
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            axes[0, 0].plot(model_data['month_date'], model_data['accuracy'], 
                           marker='o', label=f'{model} - Accuracy', linewidth=2)
            axes[0, 0].plot(model_data['month_date'], model_data['macro_f1'], 
                           marker='s', label=f'{model} - F1 Score', linewidth=2)
        
        axes[0, 0].set_title('Accuracy and F1 Score Trends')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Precision and Recall
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            axes[0, 1].plot(model_data['month_date'], model_data['precision'], 
                           marker='o', label=f'{model} - Precision', linewidth=2)
            axes[0, 1].plot(model_data['month_date'], model_data['recall'], 
                           marker='s', label=f'{model} - Recall', linewidth=2)
        
        axes[0, 1].set_title('Precision and Recall Trends')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: AUC Score
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            if 'auc' in model_data.columns and not model_data['auc'].isna().all():
                axes[1, 0].plot(model_data['month_date'], model_data['auc'], 
                               marker='o', label=model, linewidth=2)
        
        axes[1, 0].set_title('AUC Score Trends')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('AUC Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Total Samples
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            axes[1, 1].plot(model_data['month_date'], model_data['total_samples'], 
                           marker='o', label=model, linewidth=2)
        
        axes[1, 1].set_title('Total Samples Over Time')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_inference_volume_trends(self, days_back=30):
        """
        Plot model inference volume and prediction distribution over time.
        """
        df = self.get_model_inference_metrics(days_back)
        if df is None or df.empty:
            print("No inference data found.")
            return
        
        # Convert month_date to datetime
        df['month_date'] = pd.to_datetime(df['month_date'])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Total Predictions Volume
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            ax1.plot(model_data['month_date'], model_data['total_predictions'], 
                    marker='o', label=model, linewidth=2)
        
        ax1.set_title('Model Inference Volume Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Total Predictions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction Distribution (Positive vs Negative)
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            x = range(len(model_data))
            width = 0.35
            
            ax2.bar([i - width/2 for i in x], model_data['positive_predictions'], 
                   width, label=f'{model} - Positive', alpha=0.8)
            ax2.bar([i + width/2 for i in x], model_data['negative_predictions'], 
                   width, label=f'{model} - Negative', alpha=0.8)
        
        ax2.set_title('Prediction Distribution (Positive vs Negative)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Number of Predictions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_dag_success_rate(self, days_back=30):
        """
        Plot DAG success rate over time.
        """
        df = self.get_airflow_dag_metrics(days_back)
        if df is None or df.empty:
            print("No DAG execution data found.")
            return
        
        # Convert execution_date to datetime
        df['execution_date'] = pd.to_datetime(df['execution_date'])
        
        # Group by date and calculate success rate
        daily_stats = df.groupby(df['execution_date'].dt.date).agg({
            'state': lambda x: (x == 'success').sum() / len(x) * 100
        }).reset_index()
        daily_stats['execution_date'] = pd.to_datetime(daily_stats['execution_date'])
        
        # Create figure
        plt.figure(figsize=(15, 8))
        plt.plot(daily_stats['execution_date'], daily_stats['state'], 
                marker='o', linewidth=2, markersize=6, color='blue')
        plt.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='90% Success Threshold')
        plt.axhline(y=75, color='orange', linestyle='--', alpha=0.7, label='75% Success Threshold')
        
        plt.title('DAG Success Rate Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Success Rate (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison_radar(self, latest_date=None):
        """
        Create a radar chart comparing different models' performance.
        """
        df = self.get_model_performance_metrics(30)  # Last 30 days
        if df is None or df.empty:
            print("No model performance data found.")
            return
        
        # Get the latest data for each model
        if latest_date is None:
            latest_date = df['month_date'].max()
        
        latest_data = df[df['month_date'] == latest_date]
        
        if latest_data.empty:
            print(f"No data found for date: {latest_date}")
            return
        
        # Prepare data for radar chart
        metrics = ['accuracy', 'precision', 'recall', 'macro_f1']
        if 'auc' in latest_data.columns and not latest_data['auc'].isna().all():
            metrics.append('auc')
        
        # Number of variables
        N = len(metrics)
        
        # Create angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(latest_data)))
        
        for idx, (_, row) in enumerate(latest_data.iterrows()):
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['model_name'], color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics])
        ax.set_ylim(0, 1)
        
        plt.title(f'Model Performance Comparison ({latest_date.strftime("%Y-%m-%d")})', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.show()
    
    def generate_comprehensive_report(self, days_back=30):
        """
        Generate a comprehensive report with all visualizations.
        """
        print("Generating Comprehensive ML Pipeline Report...")
        print("=" * 50)
        
        # Create a large figure with all plots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. DAG Success Rate
        plt.subplot(4, 2, 1)
        df_dag = self.get_airflow_dag_metrics(days_back)
        if df_dag is not None and not df_dag.empty:
            df_dag['execution_date'] = pd.to_datetime(df_dag['execution_date'])
            daily_stats = df_dag.groupby(df_dag['execution_date'].dt.date).agg({
                'state': lambda x: (x == 'success').sum() / len(x) * 100
            }).reset_index()
            daily_stats['execution_date'] = pd.to_datetime(daily_stats['execution_date'])
            plt.plot(daily_stats['execution_date'], daily_stats['state'], 
                    marker='o', linewidth=2, color='blue')
            plt.title('DAG Success Rate', fontweight='bold')
            plt.ylabel('Success Rate (%)')
            plt.grid(True, alpha=0.3)
        
        # 2. Model Performance Trends
        plt.subplot(4, 2, 2)
        df_model = self.get_model_performance_metrics(days_back)
        if df_model is not None and not df_model.empty:
            df_model['month_date'] = pd.to_datetime(df_model['month_date'])
            for model in df_model['model_name'].unique():
                model_data = df_model[df_model['model_name'] == model]
                plt.plot(model_data['month_date'], model_data['accuracy'], 
                        marker='o', label=model, linewidth=2)
            plt.title('Model Accuracy Trends', fontweight='bold')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 3. Inference Volume
        plt.subplot(4, 2, 3)
        df_inference = self.get_model_inference_metrics(days_back)
        if df_inference is not None and not df_inference.empty:
            df_inference['month_date'] = pd.to_datetime(df_inference['month_date'])
            for model in df_inference['model_name'].unique():
                model_data = df_inference[df_inference['model_name'] == model]
                plt.plot(model_data['month_date'], model_data['total_predictions'], 
                        marker='o', label=model, linewidth=2)
            plt.title('Inference Volume', fontweight='bold')
            plt.ylabel('Total Predictions')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 4. F1 Score Trends
        plt.subplot(4, 2, 4)
        if df_model is not None and not df_model.empty:
            for model in df_model['model_name'].unique():
                model_data = df_model[df_model['model_name'] == model]
                plt.plot(model_data['month_date'], model_data['macro_f1'], 
                        marker='o', label=model, linewidth=2)
            plt.title('F1 Score Trends', fontweight='bold')
            plt.ylabel('F1 Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. DAG Execution Duration
        plt.subplot(4, 2, 5)
        if df_dag is not None and not df_dag.empty:
            successful_runs = df_dag[df_dag['state'] == 'success']
            if not successful_runs.empty:
                plt.scatter(successful_runs['execution_date'], successful_runs['duration_seconds'], 
                           alpha=0.6, s=30)
                plt.title('DAG Execution Duration', fontweight='bold')
                plt.ylabel('Duration (seconds)')
                plt.grid(True, alpha=0.3)
        
        # 6. Precision vs Recall
        plt.subplot(4, 2, 6)
        if df_model is not None and not df_model.empty:
            for model in df_model['model_name'].unique():
                model_data = df_model[df_model['model_name'] == model]
                plt.scatter(model_data['precision'], model_data['recall'], 
                           label=model, alpha=0.7, s=50)
            plt.title('Precision vs Recall', fontweight='bold')
            plt.xlabel('Precision')
            plt.ylabel('Recall')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 7. Sample Size Trends
        plt.subplot(4, 2, 7)
        if df_model is not None and not df_model.empty:
            for model in df_model['model_name'].unique():
                model_data = df_model[df_model['model_name'] == model]
                plt.plot(model_data['month_date'], model_data['total_samples'], 
                        marker='o', label=model, linewidth=2)
            plt.title('Sample Size Trends', fontweight='bold')
            plt.ylabel('Total Samples')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 8. Prediction Distribution
        plt.subplot(4, 2, 8)
        if df_inference is not None and not df_inference.empty:
            latest_inference = df_inference.groupby('model_name').last()
            if not latest_inference.empty:
                models = latest_inference.index
                positive = latest_inference['positive_predictions'].values
                negative = latest_inference['negative_predictions'].values
                
                x = np.arange(len(models))
                width = 0.35
                
                plt.bar(x - width/2, positive, width, label='Positive', alpha=0.8)
                plt.bar(x + width/2, negative, width, label='Negative', alpha=0.8)
                plt.title('Latest Prediction Distribution', fontweight='bold')
                plt.xlabel('Model')
                plt.ylabel('Number of Predictions')
                plt.xticks(x, models, rotation=45)
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("Comprehensive report generated successfully!")
        print("=" * 50)

# Example usage
if __name__ == "__main__":
    # Initialize the visualizer
    visualizer = MetricsVisualizer()
    
    # Generate individual plots
    print("Generating DAG execution timeline...")
    visualizer.plot_dag_execution_timeline(30)
    
    print("Generating model performance trends...")
    visualizer.plot_model_performance_trends(30)
    
    print("Generating inference volume trends...")
    visualizer.plot_inference_volume_trends(30)
    
    print("Generating DAG success rate...")
    visualizer.plot_dag_success_rate(30)
    
    print("Generating model comparison radar chart...")
    visualizer.plot_model_comparison_radar()
    
    print("Generating comprehensive report...")
    visualizer.generate_comprehensive_report(30) 
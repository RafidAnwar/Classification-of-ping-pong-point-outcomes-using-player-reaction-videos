import sqlite3
import pandas as pd
import os
import numpy as np

# Define the 68 pose feature names
POSE_FEATURE_NAMES = [
    'l_shoulder_x', 'l_shoulder_y', 'r_shoulder_x', 'r_shoulder_y',
    'l_elbow_x', 'l_elbow_y', 'r_elbow_x', 'r_elbow_y',
    'l_wrist_x', 'l_wrist_y', 'r_wrist_x', 'r_wrist_y',
    'l_hip_x', 'l_hip_y', 'r_hip_x', 'r_hip_y',
    'l_knee_x', 'l_knee_y', 'r_knee_x', 'r_knee_y',
    'angle_l_elbow', 'angle_r_elbow', 'angle_l_shoulder', 'angle_r_shoulder',
    'rel_lw_ls_dx', 'rel_lw_ls_dy', 'dist_lw_ls',
    'rel_rw_rs_dx', 'rel_rw_rs_dy', 'dist_rw_rs',
    'rel_le_ls_dx', 'rel_le_ls_dy', 'dist_le_ls',
    'rel_re_rs_dx', 'rel_re_rs_dy', 'dist_re_rs',
    'rel_lw_lh_dx', 'rel_lw_lh_dy', 'dist_lw_lh',
    'rel_rw_rh_dx', 'rel_rw_rh_dy', 'dist_rw_rh',
    'rel_le_lh_dx', 'rel_le_lh_dy', 'dist_le_lh',
    'rel_re_rh_dx', 'rel_re_rh_dy', 'dist_re_rh',
    'vel_l_shoulder', 'vel_r_shoulder', 'vel_l_elbow', 'vel_r_elbow',
    'vel_l_wrist', 'vel_r_wrist', 'vel_l_hip', 'vel_r_hip',
    'vel_l_knee', 'vel_r_knee',
    'acc_l_shoulder', 'acc_r_shoulder', 'acc_l_elbow', 'acc_r_elbow',
    'acc_l_wrist', 'acc_r_wrist', 'acc_l_hip', 'acc_r_hip',
    'acc_l_knee', 'acc_r_knee'
]


def explore_pose_database(db_path='./transformer/pose_features_analysis.db'):
    """
    Explore the pose features database and export for Power BI
    """

    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        print("Please run 'extract_pose_feature_importance.py' first!")
        return

    conn = sqlite3.connect(db_path)

    print("=" * 70)
    print("POSE FEATURES DATABASE EXPLORATION")
    print("=" * 70)

    # Get tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print(f"\nTables in database:")
    for table in tables:
        print(f"  - {table[0]}")

    # Export directory
    export_dir = './transformer/pose_powerbi_export/'
    os.makedirs(export_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("EXPORTING TABLES TO CSV FOR POWER BI")
    print("=" * 70)

    # Export each table
    for table_name in [t[0] for t in tables]:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        csv_path = os.path.join(export_dir, f"{table_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n{table_name}: {len(df)} records exported to {csv_path}")


    # Create combined importance table
    print(f"\n{'=' * 70}")
    print("CREATING COMBINED IMPORTANCE TABLE")
    print("=" * 70)

    combined_importance = pd.read_sql_query("""
        SELECT 
            s.feature_id,
            s.feature_name,
            s.effect_size,
            s.p_value,
            s.mean_value_win,
            s.mean_value_loss,
            s.importance_rank as stat_rank,
            g.avg_gradient_magnitude,
            g.gradient_importance_rank as gradient_rank,
            p.accuracy_drop,
            p.permutation_importance_rank as perm_rank,
            (s.importance_rank + g.gradient_importance_rank + p.permutation_importance_rank) / 3.0 as combined_rank
        FROM pose_feature_importance s
        JOIN gradient_feature_importance g ON s.feature_id = g.feature_id
        JOIN permutation_importance p ON s.feature_id = p.feature_id
        ORDER BY combined_rank
    """, conn)

    # Save combined table
    combined_path = os.path.join(export_dir, "combined_feature_importance.csv")
    combined_importance.to_csv(combined_path, index=False)
    print(f"Combined importance table saved to: {combined_path}")

    print("\nTop 15 Most Important Features (Combined Ranking):")
    print(combined_importance[['feature_id', 'feature_name', 'effect_size',
                               'avg_gradient_magnitude', 'accuracy_drop',
                               'combined_rank']].head(15).to_string(index=False))

    # Create feature correlation matrix
    print(f"\n{'=' * 70}")
    print("CREATING FEATURE CORRELATION DATA")
    print("=" * 70)

    # Get all pose features
    features_df = pd.read_sql_query("SELECT * FROM video_pose_features", conn)

    # Create correlation matrix for top 20 features
    top_20_features = combined_importance.head(20)['feature_name'].values

    corr_matrix = features_df[top_20_features].corr()

    # Convert to long format for Power BI
    corr_long = []
    for i, feat1 in enumerate(top_20_features):
        for j, feat2 in enumerate(top_20_features):
            corr_long.append({
                'feature_1_name': feat1,
                'feature_2_name': feat2,
                'correlation': corr_matrix.iloc[i, j]
            })

    corr_df = pd.DataFrame(corr_long)
    corr_path = os.path.join(export_dir, "feature_correlations.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"Feature correlation matrix saved to: {corr_path}")

    # Create feature distribution summary
    print(f"\n{'=' * 70}")
    print("CREATING FEATURE DISTRIBUTION DATA")
    print("=" * 70)

    distribution_data = []
    for feature_name in POSE_FEATURE_NAMES:
        win_data = features_df[features_df['true_label'] == 0][feature_name]
        loss_data = features_df[features_df['true_label'] == 1][feature_name]

        distribution_data.append({
            'feature_name': feature_name,
            'win_mean': win_data.mean(),
            'win_std': win_data.std(),
            'win_min': win_data.min(),
            'win_q25': win_data.quantile(0.25),
            'win_median': win_data.median(),
            'win_q75': win_data.quantile(0.75),
            'win_max': win_data.max(),
            'loss_mean': loss_data.mean(),
            'loss_std': loss_data.std(),
            'loss_min': loss_data.min(),
            'loss_q25': loss_data.quantile(0.25),
            'loss_median': loss_data.median(),
            'loss_q75': loss_data.quantile(0.75),
            'loss_max': loss_data.max(),
            'mean_difference': win_data.mean() - loss_data.mean()
        })

    dist_df = pd.DataFrame(distribution_data)
    dist_path = os.path.join(export_dir, "feature_distributions.csv")
    dist_df.to_csv(dist_path, index=False)
    print(f"Feature distribution data saved to: {dist_path}")

    conn.close()

if __name__ == "__main__":
    explore_pose_database()
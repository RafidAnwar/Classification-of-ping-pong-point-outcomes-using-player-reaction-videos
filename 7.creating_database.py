import numpy as np
import scipy.io as sio
import sqlite3
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Layer, MultiHeadAttention, LayerNormalization, Dropout, Dense
import tensorflow as tf

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


# Custom layers needed to load the model
class PositionalEncoding(Layer):
    def __init__(self, sequence_len, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_len = sequence_len
        self.embed_dim = embed_dim

    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_len': self.sequence_len,
            'embed_dim': self.embed_dim,
        })
        return config

    def build(self, input_shape):
        self.pos_encoding = self.positional_encoding(self.sequence_len, self.embed_dim)
        super(PositionalEncoding, self).build(input_shape)

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class TransformerEncoder(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


def create_pose_feature_database(db_path='./transformer/pose_features_analysis.db'):
    """Create database for pose feature analysis"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create column definitions for all 68 features
    feature_columns = ', '.join([f'{name} REAL' for name in POSE_FEATURE_NAMES])

    # Table 1: Individual pose features per video
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS video_pose_features (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            dataset_type TEXT,
            true_label INTEGER,
            predicted_label INTEGER,
            prediction_confidence REAL,
            is_correct INTEGER,
            {feature_columns}
        )
    ''')

    # Table 2: Feature importance scores
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pose_feature_importance (
            feature_id INTEGER PRIMARY KEY,
            feature_name TEXT,
            correlation_with_output REAL,
            mean_value_win REAL,
            mean_value_loss REAL,
            std_value_win REAL,
            std_value_loss REAL,
            t_statistic REAL,
            p_value REAL,
            effect_size REAL,
            importance_rank INTEGER
        )
    ''')

    # Table 3: Gradient-based feature importance
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gradient_feature_importance (
            feature_id INTEGER PRIMARY KEY,
            feature_name TEXT,
            avg_gradient_magnitude REAL,
            max_gradient_magnitude REAL,
            gradient_importance_rank INTEGER
        )
    ''')

    # Table 4: Permutation importance
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS permutation_importance (
            feature_id INTEGER PRIMARY KEY,
            feature_name TEXT,
            accuracy_drop REAL,
            permutation_importance_rank INTEGER
        )
    ''')

    conn.commit()
    return conn


def extract_pose_features_with_predictions(model, data, labels, video_ids, dataset_type, conn):
    """Extract pose features and predictions for each video"""

    pose_data, face_data = data

    # Get predictions
    predictions = model.predict([pose_data, face_data], verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    pred_confidence = predictions[:, 1]  # Confidence for class 1

    cursor = conn.cursor()

    for i, video_id in enumerate(video_ids):
        # Get average pose features across sequence (excluding padding)
        pose_seq = pose_data[i]
        valid_mask = pose_seq[:, 0] != -999

        if np.any(valid_mask):
            avg_pose_features = np.mean(pose_seq[valid_mask], axis=0)
        else:
            avg_pose_features = np.zeros(68)

        # Prepare feature values
        feature_values = [float(f) for f in avg_pose_features]

        # Create column names string
        column_names = ', '.join(POSE_FEATURE_NAMES)
        placeholders = ', '.join(['?'] * 68)

        # Insert into database
        cursor.execute(f'''
            INSERT INTO video_pose_features 
            (video_id, dataset_type, true_label, predicted_label, 
             prediction_confidence, is_correct,
             {column_names})
            VALUES (?, ?, ?, ?, ?, ?, {placeholders})
        ''', (int(video_id), dataset_type, int(labels[i]), int(pred_labels[i]),
              float(pred_confidence[i]), int(labels[i] == pred_labels[i]),
              *feature_values))

    conn.commit()
    print(f"Extracted {len(video_ids)} videos from {dataset_type} set")


def calculate_statistical_importance(conn):
    """Calculate statistical feature importance using t-tests and correlation"""
    from scipy import stats

    # Load data from database
    df = pd.read_sql_query("SELECT * FROM video_pose_features", conn)

    cursor = conn.cursor()

    for feature_idx, feature_name in enumerate(POSE_FEATURE_NAMES):
        # Separate by label
        win_values = df[df['true_label'] == 0][feature_name].values
        loss_values = df[df['true_label'] == 1][feature_name].values

        # Calculate statistics
        mean_win = np.mean(win_values)
        mean_loss = np.mean(loss_values)
        std_win = np.std(win_values)
        std_loss = np.std(loss_values)

        # T-test
        t_stat, p_value = stats.ttest_ind(win_values, loss_values)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(win_values) - 1) * std_win ** 2 + (len(loss_values) - 1) * std_loss ** 2) /
                             (len(win_values) + len(loss_values) - 2))
        effect_size = (mean_win - mean_loss) / pooled_std if pooled_std > 0 else 0

        # Correlation with output
        correlation = np.corrcoef(df[feature_name].values, df['predicted_label'].values)[0, 1]

        cursor.execute('''
            INSERT INTO pose_feature_importance 
            (feature_id, feature_name, correlation_with_output, 
             mean_value_win, mean_value_loss, std_value_win, std_value_loss,
             t_statistic, p_value, effect_size, importance_rank)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (feature_idx, feature_name, float(correlation),
              float(mean_win), float(mean_loss), float(std_win), float(std_loss),
              float(t_stat), float(p_value), float(effect_size), 0))

    # Update ranks based on absolute effect size
    cursor.execute('''
        UPDATE pose_feature_importance
        SET importance_rank = (
            SELECT COUNT(*) + 1
            FROM pose_feature_importance AS p2
            WHERE ABS(p2.effect_size) > ABS(pose_feature_importance.effect_size)
        )
    ''')

    conn.commit()
    print("Calculated statistical feature importance")


def calculate_gradient_importance(model, data, labels):
    """Calculate gradient-based feature importance"""

    pose_data, face_data = data

    # Convert to tensors
    pose_tensor = tf.constant(pose_data, dtype=tf.float32)
    face_tensor = tf.constant(face_data, dtype=tf.float32)
    labels_tensor = tf.constant(labels, dtype=tf.int32)

    gradients_list = []

    # Calculate gradients for each sample
    for i in range(len(pose_data)):
        with tf.GradientTape() as tape:
            pose_sample = tf.expand_dims(pose_tensor[i], 0)
            face_sample = tf.expand_dims(face_tensor[i], 0)
            tape.watch(pose_sample)

            predictions = model([pose_sample, face_sample], training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                labels_tensor[i:i + 1], predictions
            )

        # Get gradients with respect to pose input
        grads = tape.gradient(loss, pose_sample)
        if grads is not None:
            # Average over sequence dimension
            avg_grads = tf.reduce_mean(tf.abs(grads), axis=1)
            gradients_list.append(avg_grads.numpy()[0])

    # Average gradients across all samples
    avg_gradients = np.mean(gradients_list, axis=0)
    max_gradients = np.max(gradients_list, axis=0)

    return avg_gradients, max_gradients


def calculate_permutation_importance(model, data, labels):
    """Calculate permutation importance for pose features"""

    pose_data, face_data = data

    # Baseline accuracy
    predictions = model.predict([pose_data, face_data], verbose=0)
    baseline_acc = np.mean(np.argmax(predictions, axis=1) == labels)

    importance_scores = []

    for feature_idx in range(68):
        # Create a copy and permute one feature
        pose_permuted = pose_data.copy()

        # Permute the feature across all videos
        for i in range(len(pose_permuted)):
            valid_mask = pose_permuted[i, :, 0] != -999
            if np.any(valid_mask):
                pose_permuted[i, valid_mask, feature_idx] = np.random.permutation(
                    pose_permuted[i, valid_mask, feature_idx]
                )

        # Calculate accuracy with permuted feature
        predictions_perm = model.predict([pose_permuted, face_data], verbose=0)
        permuted_acc = np.mean(np.argmax(predictions_perm, axis=1) == labels)

        # Importance is the drop in accuracy
        importance = baseline_acc - permuted_acc
        importance_scores.append(importance)

        if (feature_idx + 1) % 10 == 0:
            print(f"Processed {feature_idx + 1}/68 features for permutation importance")

    return np.array(importance_scores)


def save_gradient_importance(conn, avg_gradients, max_gradients):
    """Save gradient-based importance to database"""

    cursor = conn.cursor()

    # Sort by average gradient magnitude
    sorted_indices = np.argsort(avg_gradients)[::-1]

    for rank, feature_idx in enumerate(sorted_indices):
        cursor.execute('''
            INSERT INTO gradient_feature_importance 
            (feature_id, feature_name, avg_gradient_magnitude, 
             max_gradient_magnitude, gradient_importance_rank)
            VALUES (?, ?, ?, ?, ?)
        ''', (int(feature_idx), POSE_FEATURE_NAMES[feature_idx],
              float(avg_gradients[feature_idx]),
              float(max_gradients[feature_idx]),
              rank + 1))

    conn.commit()
    print("Saved gradient-based importance")


def save_permutation_importance(conn, importance_scores):
    """Save permutation importance to database"""

    cursor = conn.cursor()

    # Sort by importance
    sorted_indices = np.argsort(importance_scores)[::-1]

    for rank, feature_idx in enumerate(sorted_indices):
        cursor.execute('''
            INSERT INTO permutation_importance 
            (feature_id, feature_name, accuracy_drop, permutation_importance_rank)
            VALUES (?, ?, ?, ?)
        ''', (int(feature_idx), POSE_FEATURE_NAMES[feature_idx],
              float(importance_scores[feature_idx]),
              rank + 1))

    conn.commit()
    print("Saved permutation importance")


def main():
    print("=" * 70)
    print("POSE FEATURE EXTRACTION AND ANALYSIS")
    print("=" * 70)

    # Load the trained model
    print("\n1. Loading trained model...")
    custom_objects = {
        'PositionalEncoding': PositionalEncoding,
        'TransformerEncoder': TransformerEncoder
    }
    model = load_model('./transformer/final_model.h5', custom_objects=custom_objects)
    print("Model loaded successfully!")

    # Create database
    print("\n2. Creating database...")
    conn = create_pose_feature_database()

    # Load data
    print("\n3. Loading training data...")
    data_train = sio.loadmat('trainset.mat')
    feas = data_train['features_matrix'][:, :]
    label = data_train['labels_vector'].ravel()
    video = data_train['video_tags_vector'].ravel()
    unique_videos = np.unique(video)

    pose_sequence, face_sequence, label_sequence = [], [], []
    for vid in unique_videos:
        mask = video == vid
        pose_d = feas[mask, 0:68]
        face_d = feas[mask, 68:]
        video_l = label[mask][0]
        pose_sequence.append(pose_d)
        face_sequence.append(face_d)
        label_sequence.append(video_l)

    max_length = max(seq.shape[0] for seq in pose_sequence)
    pose_data = pad_sequences(pose_sequence, maxlen=max_length, padding='post',
                              dtype='float32', value=-999)
    face_data = pad_sequences(face_sequence, maxlen=max_length, padding='post',
                              dtype='float32', value=-999)
    video_label = np.array(label_sequence)

    # Load test data
    print("\n4. Loading test data...")
    data_test = sio.loadmat('testset.mat')
    feas2 = data_test['features_matrix'][:, :]
    label2 = data_test['labels_vector'].ravel()
    video2 = data_test['video_tags_vector'].ravel()
    unique_videos_test = np.unique(video2)

    pose_sequence2, face_sequence2, label_sequence2 = [], [], []
    for vid in unique_videos_test:
        mask = video2 == vid
        pose_d2 = feas2[mask, 0:68]
        face_d2 = feas2[mask, 68:]
        video_l2 = label2[mask][0]
        pose_sequence2.append(pose_d2)
        face_sequence2.append(face_d2)
        label_sequence2.append(video_l2)

    pose_test = pad_sequences(pose_sequence2, maxlen=max_length, padding='post',
                              dtype='float32', value=-999)
    face_test = pad_sequences(face_sequence2, maxlen=max_length, padding='post',
                              dtype='float32', value=-999)
    y_test = np.array(label_sequence2)

    # Extract features and predictions
    print("\n5. Extracting pose features with predictions...")
    extract_pose_features_with_predictions(
        model, (pose_data, face_data), video_label, unique_videos, 'train', conn
    )
    extract_pose_features_with_predictions(
        model, (pose_test, face_test), y_test, unique_videos_test, 'test', conn
    )

    # Calculate statistical importance
    print("\n6. Calculating statistical feature importance...")
    calculate_statistical_importance(conn)

    # Calculate gradient-based importance
    print("\n7. Calculating gradient-based importance (this may take a few minutes)...")
    avg_grads, max_grads = calculate_gradient_importance(
        model, (pose_test, face_test), y_test
    )
    save_gradient_importance(conn, avg_grads, max_grads)

    # Calculate permutation importance
    print("\n8. Calculating permutation importance (this will take several minutes)...")
    perm_scores = calculate_permutation_importance(
        model, (pose_test, face_test), y_test
    )
    save_permutation_importance(conn, perm_scores)

    conn.close()

    print("\n" + "=" * 70)
    print("Database saved to: ./transformer/pose_features_analysis.db")
    print("=" * 70)


if __name__ == "__main__":
    main()
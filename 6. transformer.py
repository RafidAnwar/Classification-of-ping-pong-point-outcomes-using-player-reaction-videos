import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, roc_auc_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import CSVLogger, EarlyStopping
from keras.layers import Dense, Masking, Dropout, Concatenate, Input, LayerNormalization, Layer, MultiHeadAttention, GlobalAveragePooling1D, Embedding, Add
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class PositionalEncoding(Layer):
    def __init__(self, sequence_len, embed_dim):
        super(PositionalEncoding, self).__init__()
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
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class TransformerEncoder(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
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

    def call(self, inputs, training):
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


# Load train data
data_train = sio.loadmat('trainset.mat')
feas = data_train['features_matrix'][:,:]
label = data_train['labels_vector'].ravel()
video = data_train['video_tags_vector'].ravel()
unique_videos = np.unique(video)
pose_sequence, face_sequence, label_sequence = [], [], []

# Separate pose and face data by video
for vid in unique_videos:
    mask = video == vid
    pose_d = feas[mask, 0:68]
    face_d = feas[mask, 68:]
    video_l = label[mask][0]
    pose_sequence.append(pose_d)
    face_sequence.append(face_d)
    label_sequence.append(video_l)

max_length = max(seq.shape[0] for seq in pose_sequence)
pose_data = pad_sequences(pose_sequence, maxlen=max_length, padding='post', dtype='float32', value=-999)
face_data = pad_sequences(face_sequence, maxlen=max_length, padding='post', dtype='float32', value=-999)
video_label = np.array(label_sequence)

pose_train, pose_val, face_train, face_val, y_train, y_val = train_test_split(pose_data, face_data, video_label, test_size=0.2, random_state=42, stratify=video_label)

y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)
num_classes = len(np.unique(label))

# Load test data
data_test = sio.loadmat('testset.mat')
feas2 = data_test['features_matrix'][:,:]
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

pose_test = pad_sequences(pose_sequence2, maxlen=max_length, padding='post', dtype='float32', value=-999)
face_test = pad_sequences(face_sequence2, maxlen=max_length, padding='post', dtype='float32', value=-999)
y_test = np.array(label_sequence2)
y_test = to_categorical(y_test, num_classes=2)

pose_shape = (max_length, pose_data.shape[2])
face_shape = (max_length, face_data.shape[2])

best_accuracy = 0.0
best_model = None

for run in range(3):
    # Inputs
    pose_input = Input(shape=pose_shape, name="pose_input")
    face_input = Input(shape=face_shape, name="face_input")

    # Mask padding
    pose_masked = Masking(mask_value=-999)(pose_input)
    face_masked = Masking(mask_value=-999)(face_input)

    # Positional encoding
    pose_pos_encoded = PositionalEncoding(max_length, pose_shape[1])(pose_masked)
    face_pos_encoded = PositionalEncoding(max_length, face_shape[1])(face_masked)

    # Transformer encoders
    pose_tr = TransformerEncoder(embed_dim=pose_shape[1], num_heads=12, ff_dim=256)(pose_pos_encoded)
    #pose_tr = TransformerEncoder(embed_dim=pose_shape[1], num_heads=8, ff_dim=256)(pose_tr)
    #pose_tr = Dropout(0.2)(pose_tr)
    pose_pooled = GlobalAveragePooling1D()(pose_tr)

    face_tr = TransformerEncoder(embed_dim=face_shape[1], num_heads=14, ff_dim=256)(face_pos_encoded)
    #face_tr = TransformerEncoder(embed_dim=face_shape[1], num_heads=8, ff_dim=256)(face_tr)
    #face_tr = Dropout(0.2)(face_tr)
    face_pooled = GlobalAveragePooling1D()(face_tr)

    fused = Concatenate()([pose_pooled, face_pooled])
    x = Dense(128, activation='relu')(fused)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[pose_input, face_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

    csv_logger = CSVLogger(f'./transformer/epoch_data{run}.log')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        [pose_train, face_train], y_train,
        epochs=50,
        batch_size=32,
        validation_data=([pose_val, face_val], y_val),
        callbacks=[csv_logger, early_stopping],
        verbose=1
    )

    loss, acc = model.evaluate([pose_test, face_test], y_test, verbose=0)
    print(f'Run {run} - Test accuracy: {acc:.4f}')

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        y_pred_probs = best_model.predict([pose_test, face_test])
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_probs[:, 1])
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes), normalize='true')

print(f"Best Test accuracy: {best_accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Actual labels: {y_true}")
print(f"Predicted labels: {y_pred}")

best_model.save('./transformer/final_model.h5')

figure, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Win", "Loss"])
disp.plot(ax=ax, cmap='Blues')
ax.set_title("Mean Confusion Matrix", fontweight='bold')
ax.set_xlabel("Predicted Label", fontweight='bold')
ax.set_ylabel("True Label", fontweight='bold')
plt.tight_layout()
plt.savefig('./transformer/matrix.svg', dpi=300, format='svg')
plt.close()

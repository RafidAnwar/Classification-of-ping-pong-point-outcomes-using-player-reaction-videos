import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score,precision_score, roc_auc_score
#import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import CSVLogger
from keras.layers import LSTM, Dense, Masking, Dropout, Concatenate, Input
from keras.regularizers import l1_l2
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# training data
data = sio.loadmat('trainset.mat')
feas = data['features_matrix'][:,:]
label = data['labels_vector'].ravel()
video = data['video_tags_vector'].ravel()
unique_videos = np.unique(video)
pose_sequence,face_sequence, label_sequence = [], [], []

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

pose_train, pose_val, face_train, face_val, y_train, y_val = train_test_split(pose_data,face_data, video_label, test_size=0.2, random_state=42, stratify=video_label)

y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)
num_classes = len(np.unique(label))

# Test data
data2 = sio.loadmat('testset.mat')
feas2 = data2['features_matrix'][:,:]
label2 = data2['labels_vector'].ravel()
video2 = data2['video_tags_vector'].ravel()
unique_videos = np.unique(video2)
pose_sequence2, face_sequence2, label_sequence2 = [], [], []
for vid in unique_videos:
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
    # 2 layered LSTM model architecture

    # pose branch
    pose_input = Input(shape=pose_shape, name='pose_input')
    pose_masked = Masking(mask_value=-999)(pose_input)
    pose_lstm = LSTM(64,kernel_regularizer=l1_l2(l1=0.005,l2=.01), return_sequences=False )(pose_masked)
    pose_lstm = Dropout(0.1)(pose_lstm)
    #pose_lstm = LSTM(128, kernel_regularizer=l1(l1=0.005))(pose_lstm)'''

    #face branch
    face_input = Input(shape=face_shape, name='face_input')
    face_masked = Masking(mask_value=-999)(face_input)
    face_lstm = LSTM(64,kernel_regularizer=l1_l2(l1=0.005,l2=.01), return_sequences=False )(face_masked)
    face_lstm = Dropout(0.1)(face_lstm)
    #face_lstm = LSTM(128, kernel_regularizer=l1(l1=0.005))(face_lstm)'''

    merged = Concatenate()([pose_lstm, face_lstm])
    dense1 = Dense(64, activation='relu')(merged)
    dense = Dropout(0.1)(dense1)
    output = Dense(num_classes, activation='softmax')(dense)

    model = Model(inputs=[pose_input, face_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate= 0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

    # Create the CSVLogger callback to store per epoch results
    csv_logger = CSVLogger(f'./lstm/epoch_data{run}.log')
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    # Train the model
    model.fit(
        [pose_train, face_train], y_train,
        epochs=40,
        batch_size=32,
        validation_data=([pose_val, face_val], y_val),
        callbacks=[csv_logger, early_stopping],
        verbose=1
    )
    # Evaluate model with metrics
    loss, acc = model.evaluate([pose_test, face_test], y_test, verbose=0)
    print(f' run{run} : test accuracy {acc}')

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        y_pred_probs = best_model.predict([pose_test, face_test])
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_probs[:,1])
        cm = confusion_matrix(y_true, y_pred, labels=range(2), normalize='true')


print(f"Test accuracy: {best_accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"AUC: {auc:.4f}")
print(f'actual labels: {y_true}')
print(f'predicted labels: {y_pred}')

# save the model to be used later for explainability analysis
best_model.save(f'./lstm/final_model.h5')

# Compute and display the mean confusion matrix across all participants
figure, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Win", "Loss"])
disp.plot(ax=ax, cmap='Blues')
ax.set_title("Mean Confusion Matrix", fontweight='bold')
ax.set_xlabel("Predicted Label", fontweight='bold')
ax.set_ylabel("True Label", fontweight='bold')
plt.tight_layout()
plt.savefig(f'./lstm/matrix.svg', dpi=300, format='svg')
plt.close()
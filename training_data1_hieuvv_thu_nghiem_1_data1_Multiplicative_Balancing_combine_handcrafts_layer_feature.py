import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from scipy.interpolate import make_interp_spline
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from collections import Counter
from sklearn.model_selection import train_test_split

def get_callbacks(result_dir):
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,  # Giảm learning rate nếu val_loss không cải thiện sau 3 epoch
        min_lr=1e-6,
        verbose=1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(result_dir, 'best_weights.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    return [reduce_lr, checkpoint]


def create_keras_model(X_train, y_train):
    model = Sequential()

    # Input layer
    model.add(Dense(512, input_dim=X_train.shape[1], kernel_regularizer=l2(0.001)))  # Tăng số lượng nơ-ron
    model.add(BatchNormalization())
    model.add(Activation('relu'))  # Sử dụng ReLU
    model.add(Dropout(0.3))  # Tăng Dropout

    # Hidden layers
    model.add(Dense(256, kernel_regularizer=l2(0.001)))  # Thêm tầng ẩn lớn hơn
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(128, kernel_regularizer=l2(0.001)))  # Giữ nguyên kích thước tầng ẩn
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(64, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))  # Tăng Dropout nhẹ

    # Output layer
    model.add(Dense(y_train.shape[1], activation='softmax'))

    # Compile model using Adam optimizer with learning rate scheduler
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)   # Tăng learning rate và dùng AdamW
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model



def normalize_data(train_data, test_data):
    """
    Normalize the data using StandardScaler.
    """
    scaler = StandardScaler()
    train_data_normalized = scaler.fit_transform(train_data)
    test_data_normalized = scaler.transform(test_data)
    return train_data_normalized, test_data_normalized


def plot_combined_metrics(metric_collection, result_dir):
    """
    Plot combined Precision, Recall, F1-Score, Sensitivity, and Specificity for all models.
    Each batch size will have its own chart.
    """
    df = pd.DataFrame(metric_collection)

    # List of metrics to plot
    metrics = ["Precision", "Recall", "F1 Score", "Sensitivity", "Specificity", 
           "Best Validation Accuracy", "Test Accuracy", "Time Taken"]

    metric_titles = {
        "Precision": "Precision Comparison",
        "Recall": "Recall Comparison",
        "F1 Score": "F1-Score Comparison",
        "Sensitivity": "Sensitivity Comparison",
        "Specificity": "Specificity Comparison",
        "Best Validation Accuracy": "Validation Accuracy Comparison",  # Sửa đổi ở đây
        "Test Accuracy": "Test Accuracy Comparison",
        "Time Taken": "Training Time Comparison"
    }

    
    # Define colors and patterns for bars
    colors = plt.cm.tab10.colors  # Use a colormap with distinct colors
    patterns = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']  # Different bar patterns

    # Group by batch size
    batch_sizes = df["Batch Size"].unique()
    for batch_size in batch_sizes:
        df_batch = df[df["Batch Size"] == batch_size]
        
        batch_folder = os.path.join(result_dir, f"batch_size_{batch_size}")
        os.makedirs(batch_folder, exist_ok=True)
        
        for metric in metrics:
            if metric not in df_batch.columns:
                print(f"Metric '{metric}' not found in dataset. Skipping.")
                continue
      
            plt.figure(figsize=(14, 8))

            # Prepare data for plotting
            grouped_data = df_batch.groupby(["Model"])[metric].mean().reset_index()
            models = grouped_data["Model"].unique()

            bar_width = 0.5  # Width of each bar
            x_positions = np.arange(len(models))  # X-axis positions for models

            # Plot bars for each model
            for i, model in enumerate(models):
                model_value = grouped_data[grouped_data["Model"] == model][metric].values[0]
                plt.bar(
                    x_positions[i],
                    model_value,
                    bar_width,
                    label=f'{model}',
                    color=colors[i % len(colors)],
                    hatch=patterns[i % len(patterns)]
                )

                # Add value annotations at the top of each bar
                plt.text(
                    x_positions[i],
                    model_value + 0.01,
                    f'{model_value:.2f}',
                    ha='center',
                    fontsize=10,
                    color='black'
                )
                
            # Remove x-axis tick labels
            plt.xticks(x_positions, [''] * len(models))  # Set empty strings for x-axis ticks
            # Set x-axis labels and legend
            # plt.xticks(x_positions, models, rotation=45, ha='right')  # Rotate model names for readability
            plt.ylabel(metric)
            # plt.title(f'{metric_titles[metric]} (Batch Size: {batch_size})')
            plt.legend(loc='upper left', title="Models", fontsize=10)
            plt.tight_layout()

            # Save the plot
            plt.savefig(os.path.join(batch_folder, f'{metric.lower().replace(" ", "_")}_batch_size_{batch_size}_comparison.png'))
            plt.close()

    print("All combined metric comparison plots saved.")


def plot_epoch_based_metrics(all_histories, result_dir):
    """
    Vẽ biểu đồ timeline của Train Loss, Validation Loss, Train Accuracy, Validation Accuracy
    theo các giá trị batch_size.
    """
    # Convert `all_histories` dictionary to a DataFrame
    metrics_list = []
    for model_name, model_histories in all_histories.items():
        for history_entry in model_histories:
            batch_size = history_entry["batch_size"]
            epoch = history_entry["epoch"]
            history = history_entry["history"]
            
            for epoch_idx, (train_loss, val_loss, train_acc, val_acc) in enumerate(
                zip(history["loss"], history["val_loss"], history["accuracy"], history["val_accuracy"])
            ):
                metrics_list.append({
                    "Model": model_name,
                    "Batch Size": batch_size,
                    "Epoch": epoch_idx + 1,
                    "Train Loss": train_loss,
                    "Validation Loss": val_loss,
                    "Train Accuracy": train_acc,
                    "Validation Accuracy": val_acc,
                })

    # Convert to DataFrame
    df = pd.DataFrame(metrics_list)

    # Metrics cần vẽ
    metrics = ["Train Loss", "Validation Loss", "Train Accuracy", "Validation Accuracy"]

    # Lặp qua từng batch size
    batch_sizes = df["Batch Size"].unique()
    for batch_size in batch_sizes:
        batch_folder = os.path.join(result_dir, f"batch_size_{batch_size}")
        os.makedirs(batch_folder, exist_ok=True)

        for metric in metrics:
            plt.figure(figsize=(14, 8))
            batch_df = df[df["Batch Size"] == batch_size]
            for model_name, model_df in batch_df.groupby("Model"):
                epochs = model_df["Epoch"].values
                metric_values = model_df[metric].values

                # Vẽ đường timeline cho mỗi mô hình
                plt.plot(epochs, metric_values, label=model_name, marker='o', linestyle='-')

            plt.xlabel("Epochs", fontsize=12)
            plt.ylabel(metric, fontsize=12)
            # plt.title(f"{metric} Timeline Comparison Across Models (Batch Size: {batch_size})", fontsize=14)
            plt.grid(alpha=0.3)
            plt.legend(title="Models", loc="best", fontsize=10)
            plt.tight_layout()

            # Lưu biểu đồ
            plot_path = os.path.join(batch_folder, f"{metric.lower().replace(' ', '_')}_batch_size_{batch_size}_timeline_comparison.png")
            plt.savefig(plot_path)
            plt.close()

    print(f"Epoch-based timeline comparison plots saved.")
      
# def plot_all_figures(batch_size, epoch, history, y_true_labels, y_pred_labels, y_pred_probs, categories, result_out, model_name):
def plot_all_figures(batch_size, epoch, history, y_true_labels, y_pred_labels, y_pred_probs, categories, result_out, model_name):
    """
    Plots Accuracy, Loss, Confusion Matrix, ROC Curve, and Accuracy vs. Recall plots.
    """
    # 1. Accuracy Plot
    plt.figure()
    plt.plot(history.history['accuracy'], linestyle='--')
    plt.plot(history.history['val_accuracy'], linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_accuracy_plot.png'))
    plt.close()

    # 2. Loss Plot
    plt.figure()
    plt.plot(history.history['loss'], linestyle='--')
    plt.plot(history.history['val_loss'], linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_loss_plot.png'))
    plt.close()

    # 3. Confusion Matrix Plot with Float Numbers
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_confusion_matrix_normalized.png'))
    plt.close()

    # Encode the true labels to binary format
    label_encoder = LabelEncoder()
    y_true_binary = label_encoder.fit_transform(y_true_labels)

    # 4. ROC Curve Plot for each class in a one-vs-rest fashion
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10.colors  # Simplified colors
    line_styles = ['-', '--', '-.', ':']  # Updated line styles
    line_width = 1.5  # Reduced line thickness

    # Ensure y_true_labels and y_pred_labels are NumPy arrays and encode labels if they are not integers
    label_encoder = LabelEncoder()
    if isinstance(y_true_labels[0], str) or isinstance(y_true_labels[0], bool):
        y_true_labels = label_encoder.fit_transform(y_true_labels)
    else:
        y_true_labels = np.array(y_true_labels)

    if isinstance(y_pred_labels[0], str) or isinstance(y_pred_labels[0], bool):
        y_pred_labels = label_encoder.transform(y_pred_labels)
    else:
        y_pred_labels = np.array(y_pred_labels)

    if len(categories) == 2:  # Binary classification case
        # Plotting for the positive class (1)
        y_true_binary = (y_true_labels == 1).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[1], linestyle=line_styles[0], linewidth=line_width, label=f'{categories[1]} (AUC = {roc_auc:.4f})')
        
        # Plotting for the negative class (0)
        y_true_binary = (y_true_labels == 0).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, 0])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[0], linestyle=line_styles[1], linewidth=line_width, label=f'{categories[0]} (AUC = {roc_auc:.4f})')
        
    else:  # Multi-class case
        for i, class_name in enumerate(categories):
            y_true_binary = (y_true_labels == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr,
                color=colors[i % len(colors)],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=line_width,
                label=f'{class_name} (AUC = {roc_auc:.4f})'
            )

    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label="Chance (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multiple Classes')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_roc_curve.png'))
    plt.close()

    # 5. Accuracy vs. Recall Plot
    report = classification_report(y_true_labels, y_pred_labels, target_names=categories, output_dict=True)
    accuracy = [report[category]['precision'] for category in categories]
    recall = [report[category]['recall'] for category in categories]

    plt.figure()
    plt.plot(categories, accuracy, marker='o', linestyle='--', color='b', label='Accuracy')
    plt.plot(categories, recall, marker='o', linestyle='-', color='g', label='Recall')
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.legend(loc='best')
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_accuracy_vs_recall.png'))
    plt.close()

    print(f"All plots saved to {result_out}")

    # 6. Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    if len(categories) == 2:  # Binary classification case
        # Plotting for the positive class (1)
        y_true_binary = (y_true_labels == 1).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_probs[:, 1])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color=colors[1], linestyle=line_styles[0], linewidth=line_width, 
                 label=f'{categories[1]} (PR AUC = {pr_auc:.4f})')

        # Plotting for the negative class (0)
        y_true_binary = (y_true_labels == 0).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_probs[:, 0])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color=colors[0], linestyle=line_styles[1], linewidth=line_width, 
                 label=f'{categories[0]} (PR AUC = {pr_auc:.4f})')
    else:  # Multi-class case
        for i, class_name in enumerate(categories):
            y_true_binary = (y_true_labels == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_probs[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(
                recall, precision,
                color=colors[i % len(colors)],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=line_width,
                label=f'{class_name} (PR AUC = {pr_auc:.4f})'
            )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_precision_recall_curve.png'))
    plt.close()
    
    
def load_features_with_smote(layer_paths, labels):
    """
    Load features for a specific layer and apply SMOTE to balance classes.

    Args:
        layer_paths (dict): Dictionary with categories as keys and paths as values.
        labels (list): List of category names.

    Returns:
        np.ndarray: Resampled features (X_resampled).
        np.ndarray: Resampled labels (y_resampled).
    """
    X = []
    y = []

    for label in labels:
        if label in layer_paths:
            label_folder = layer_paths[label]
            if os.path.isdir(label_folder):
                for filename in os.listdir(label_folder):
                    if filename.endswith('.npy'):
                        feature_path = os.path.join(label_folder, filename)
                        feature = np.load(feature_path)
                        X.append(feature)
                        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Print the distribution of labels
    from collections import Counter
    label_distribution = Counter(y)
    print(f"Label distribution: {dict(label_distribution)}")
    
    # Encode labels to numerical format for SMOTE
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Apply SMOTE
    print(f"Before SMOTE: Class distribution = {np.bincount(y_encoded)}")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
    print(f"After SMOTE: Class distribution = {np.bincount(y_resampled)}")

    return X_resampled, label_encoder.inverse_transform(y_resampled)

def load_features_without_smote(layer_paths, labels):
    """
    Load features for a specific layer without applying SMOTE.
    
    Args:
        layer_paths (dict): Dictionary with categories as keys and paths as values.
        labels (list): List of category names.

    Returns:
        np.ndarray: Features (X).
        np.ndarray: Corresponding labels (y).
    """
    X = []
    y = []

    for label in labels:
        if label in layer_paths:
            label_folder = layer_paths[label]
            if os.path.isdir(label_folder):
                for filename in os.listdir(label_folder):
                    if filename.endswith('.npy'):
                        feature_path = os.path.join(label_folder, filename)
                        feature = np.load(feature_path)
                        X.append(feature)
                        y.append(label)
    
    # Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Print the distribution of labels
    from collections import Counter
    label_distribution = Counter(y)
    print(f"Label distribution (without SMOTE): {dict(label_distribution)}")

    return X, y

def run_layer_experiment(epoch_values, batch_size_list, metric_collection, train_paths, test_paths, val_paths, layers, result_dir, categories):
    performance_metrics = []
    all_histories = {}

    for layer_name in layers:
        print(f"Running experiment for layer: {layer_name}")

        all_histories[layer_name] = []
        layer_result_out = os.path.join(result_dir, layer_name)
        os.makedirs(layer_result_out, exist_ok=True)

        train_layer_paths = {cat: train_paths[layer_name][cat] for cat in categories}
        val_layer_paths = {cat: val_paths[layer_name][cat] for cat in categories}
        test_layer_paths = {cat: test_paths[layer_name][cat] for cat in categories}

        train_images, train_labels = load_features_without_smote(train_layer_paths, categories)
        val_images, val_labels = load_features_without_smote(val_layer_paths, categories)
        test_images, test_labels = load_features_without_smote(test_layer_paths, categories)

        # Phân bổ lại dữ liệu: 70% train, 15% val, 15% test
        all_images = np.concatenate([train_images, val_images, test_images], axis=0)
        all_labels = np.concatenate([train_labels, val_labels, test_labels], axis=0)

        train_images, temp_images, train_labels, temp_labels = train_test_split(
            all_images, all_labels, test_size=0.3, stratify=all_labels, random_state=42
        )
        val_images, test_images, val_labels, test_labels = train_test_split(
            temp_images, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
        )

        # Áp dụng SMOTE để cân bằng dữ liệu train
        print(f"Before SMOTE: Class distribution = {Counter(train_labels)}")
        smote = SMOTE(random_state=42)
        train_images, train_labels = smote.fit_resample(train_images.reshape(train_images.shape[0], -1), train_labels)
        print(f"After SMOTE: Class distribution = {Counter(train_labels)}")

        train_images = train_images.reshape(train_images.shape[0], -1)
        val_images = val_images.reshape(val_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)

        train_images_normalized, test_images_normalized = normalize_data(train_images, test_images)
        _, val_images_normalized = normalize_data(train_images, val_images)

        print(f"Train images (normalized): {train_images_normalized.shape}")
        print(f"Validation images (normalized): {val_images_normalized.shape}")
        print(f"Test images (normalized): {test_images_normalized.shape}")

        label_encoder = LabelEncoder()
        train_labels_encoded = label_encoder.fit_transform(train_labels)
        val_labels_encoded = label_encoder.transform(val_labels)
        test_labels_encoded = label_encoder.transform(test_labels)

        num_categories = len(np.unique(train_labels_encoded))
        train_labels_categorical = to_categorical(train_labels_encoded, num_categories)
        val_labels_categorical = to_categorical(val_labels_encoded, num_categories)
        test_labels_categorical = to_categorical(test_labels_encoded, num_categories)

        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels_encoded), y=train_labels_encoded)
        class_weights_dict = dict(enumerate(class_weights))

        print(f"Training data: {train_images_normalized.shape}, Training labels: {train_labels_categorical.shape}")
        print(f"Validation data: {val_images_normalized.shape}, Validation labels: {val_labels_categorical.shape}")
        print(f"Class weights: {class_weights_dict}")

        for batch_size in batch_size_list:
            batch_size_result_out = os.path.join(layer_result_out, f'batch_size_{batch_size}')
            os.makedirs(batch_size_result_out, exist_ok=True)

            for epoch in epoch_values:
                epoch_result_out = os.path.join(batch_size_result_out, f'epoch_{epoch}')
                os.makedirs(epoch_result_out, exist_ok=True)

                model = create_keras_model(train_images_normalized, train_labels_categorical)

                best_weights_path = os.path.join(epoch_result_out, 'best_weights.h5')
                callbacks = get_callbacks(epoch_result_out)

                start_time = time.time()
                history = model.fit(
                    train_images_normalized,
                    train_labels_categorical,
                    epochs=epoch,
                    batch_size=batch_size,
                    validation_data=(val_images_normalized, val_labels_categorical),
                    class_weight=class_weights_dict,
                    callbacks=callbacks,
                    verbose=2
                )
                end_time = time.time()
                time_taken = end_time - start_time

                model.load_weights(best_weights_path)

                best_epoch = np.argmax(history.history['val_accuracy']) + 1
                best_val_accuracy = history.history['val_accuracy'][best_epoch - 1]

                y_pred_probs = model.predict(test_images_normalized)
                y_pred_labels = np.argmax(y_pred_probs, axis=1)
                y_true_labels = np.argmax(test_labels_categorical, axis=1)

                test_accuracy = accuracy_score(y_true_labels, y_pred_labels)
                precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
                recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
                f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
                cm = confusion_matrix(y_true_labels, y_pred_labels)

                sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
                specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0

                performance_metrics.append({
                    "Model": layer_name,
                    "Layer": layer_name,
                    "Batch Size": batch_size,
                    "Epoch": epoch,
                    "Best Epoch": best_epoch,
                    "Best Validation Accuracy": best_val_accuracy,
                    "Test Accuracy": test_accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "Sensitivity": sensitivity,
                    "Specificity": specificity,
                    "Time Taken": time_taken
                })

                all_histories[layer_name].append({
                    "batch_size": batch_size,
                    "epoch": epoch,
                    "history": {
                        "loss": history.history['loss'],
                        "val_loss": history.history['val_loss'],
                        "accuracy": history.history.get('accuracy', []),
                        "val_accuracy": history.history.get('val_accuracy', []),
                    },
                })

                plot_all_figures(batch_size, epoch, history, y_true_labels, y_pred_labels, y_pred_probs, categories, epoch_result_out, layer_name)

    metric_collection.extend(performance_metrics)
    performance_df = pd.DataFrame(performance_metrics)
    performance_df.to_csv(os.path.join(result_dir, 'performance_metrics.csv'), index=False)

    print(f"Performance metrics saved to {os.path.join(result_dir, 'performance_metrics.csv')}")
    return all_histories, metric_collection



# Function to dynamically generate train/test/val paths
def generate_paths(feature_dir, dataset_type, categories, layers):
    paths = {}
    for category in categories:
        category_paths = {}
        for layer in layers:
            layer_path = os.path.join(
                feature_dir, f"{dataset_type}/layer_handcrafts_fc2_features/{category}/{layer}"
            )
            category_paths[layer] = layer_path
        paths[category] = category_paths
    return paths

def main():
    # Directory for the BlockNet dataset
    base_dir = os.getcwd()
    home_dir = os.path.join(base_dir, 'data1') 
    feature_dir = os.path.join(home_dir, 'data1_Multiplicative_Balancing_combine_handcrafts_layer_feature')
    result_dir = os.path.join(home_dir, 'training_data1_hieuvv_thu_nghiem_1_data1_Multiplicative_Balancing_combine_handcrafts_layer_feature')
    os.makedirs(result_dir, exist_ok=True)

    # Define categories and layers
    categories = ['BCC', 'MM', 'SCC']
    # categories = ['BCC', 'MM', 'SCC', 'no skin cancer']
    num_classes = len(categories)

    batch_size_list = [8, 16, 32, 64, 128]
    epoch_values = [200]
    
    # batch_size_list = [8]
    # epoch_values = [50]
    
    metric_collection = []

    print("Generating paths for layers and categories...")
    layers = [
        'block1_conv1', 'block1_conv2',
        'block2_conv1', 'block2_conv2',
        'block3_conv1', 'block3_conv2', 'block3_conv3',
        'block4_conv1', 'block4_conv2', 'block4_conv3',
        'block5_conv1', 'block5_conv2', 'block5_conv3'
    ]

    # layers = ['block1_conv1', 'block1_conv2',]
    
    # Generate train, val, and test paths
    train_paths = generate_paths(feature_dir, "train", categories, layers)
    val_paths = generate_paths(feature_dir, "val", categories, layers)
    test_paths = generate_paths(feature_dir, "test", categories, layers)

    print("Paths generated successfully. Preparing for experiments...")

    # Flatten paths by layers for experiments
    flattened_train_paths = {layer: {cat: train_paths[cat][layer] for cat in categories} for layer in layers}
    flattened_val_paths = {layer: {cat: val_paths[cat][layer] for cat in categories} for layer in layers}
    flattened_test_paths = {layer: {cat: test_paths[cat][layer] for cat in categories} for layer in layers}

    print("Starting experiments for each layer...")

    # Run experiments for each layer
    all_histories, metric_collection = run_layer_experiment(
        epoch_values=epoch_values,
        batch_size_list=batch_size_list,
        metric_collection=metric_collection,
        train_paths=flattened_train_paths,
        test_paths=flattened_test_paths,
        val_paths=flattened_val_paths,
        layers=layers,
        result_dir = result_dir,
        categories = categories
    )

    print("Experiments completed. Saving results...")

    # Save performance metrics to a CSV file
    metric_collection_df = pd.DataFrame(metric_collection)
    metric_collection_df.to_csv(os.path.join(result_dir, 'overall_performance_metrics.csv'), index=False)
    print(f"Overall performance metrics saved to {os.path.join(result_dir, 'overall_performance_metrics.csv')}")

    # Generate plots for the results
    print("Generating combined metric plots...")
    plot_combined_metrics(metric_collection, result_dir)

    print("Generating epoch-based timeline plots...")
    plot_epoch_based_metrics(all_histories, result_dir)

    print("All tasks completed successfully.")


if __name__ == "__main__":
    main()


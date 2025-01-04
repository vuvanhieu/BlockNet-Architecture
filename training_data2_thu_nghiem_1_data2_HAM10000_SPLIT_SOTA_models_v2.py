import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16, VGG19, ResNet152, InceptionV3, EfficientNetB0, EfficientNetB7, MobileNetV2, Xception, DenseNet121
from tensorflow.keras.utils import plot_model
from keras_vit.vit import ViT_B32
from tensorflow.keras.utils import get_file
from vit_keras import vit, utils
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import make_interp_spline



def calculate_class_weights(train_labels):
    # Tính toán class weights sử dụng sklearn
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',  # Sử dụng chiến lược cân bằng
        classes=np.unique(train_labels),  # Các lớp độc nhất trong nhãn
        y=train_labels  # Nhãn của tập huấn luyện
    )

    # Chuyển class weights thành dictionary
    class_weight_dict = dict(enumerate(class_weights))

    return class_weight_dict

def save_class_weights_to_txt(class_weight_dict, save_path):
    with open(save_path, 'w') as f:
        for class_index, weight in class_weight_dict.items():
            f.write(f"Class {class_index}: Weight {weight}\n")
    print(f"Class weights saved to {save_path}")

# Learning rate schedule
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    elif 10 <= epoch < 30:
        return lr * 0.1
    else:
        return lr * 0.01

# Save parameters function
def save_parameters(model_name, batch_size, epoch, model, save_path):
    # Convert the learning rate to a standard Python float
    optimizer_config = model.optimizer.get_config()
    optimizer_config['learning_rate'] = float(optimizer_config['learning_rate'])

    # Convert all other float32 values in optimizer_config to float
    for key, value in optimizer_config.items():
        if isinstance(value, np.float32):
            optimizer_config[key] = float(value)

    # Define the TXT file path
    txt_file_path = os.path.join(save_path, f'{model_name}_bs{batch_size}_ep{epoch}_params.txt')

    # Write parameters to TXT
    with open(txt_file_path, 'w') as txt_file:
        txt_file.write(f"Model Name: {model_name}\n")
        txt_file.write(f"Batch Size: {batch_size}\n")
        txt_file.write(f"Epochs: {epoch}\n")
        txt_file.write(f"Initial Learning Rate: {optimizer_config['learning_rate']}\n")
        txt_file.write("Learning Rate Schedule: Decrease after 10 and 30 epochs\n")
        txt_file.write("Optimizer Configuration:\n")
        for key, value in optimizer_config.items():
            txt_file.write(f"  {key}: {value}\n")
        txt_file.write("Model Architecture:\n")
        txt_file.write(model.to_json())

    print(f"Model parameters saved to {txt_file_path}")
    

def plot_combined_metrics(metric_collection, result_folder):
    """
    Plot combined Precision, Recall, F1-Score, Sensitivity, and Specificity for all models.
    Each batch size will have its own chart.
    """
    df = pd.DataFrame(metric_collection)

    # List of metrics to plot
    metrics = [
        "Precision", "Recall", "F1 Score", "Sensitivity", "Specificity",
        "Best Validation Accuracy", "Test Accuracy", "Time Taken"
    ]
    metric_titles = {
        "Precision": "Precision Comparison",
        "Recall": "Recall Comparison",
        "F1 Score": "F1-Score Comparison",
        "Sensitivity": "Sensitivity Comparison",
        "Specificity": "Specificity Comparison",
        "Best Validation Accuracy": "Best Validation Accuracy Comparison",
        "Test Accuracy": "Test Accuracy Comparison",
        "Time Taken": "Training Time Comparison (Seconds)"
    }

    # Define colors and patterns for bars
    colors = plt.cm.tab10.colors  # Use a colormap with distinct colors
    patterns = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']  # Different bar patterns

    # Group by batch size
    batch_sizes = df["Batch Size"].unique()
    for batch_size in batch_sizes:
        df_batch = df[df["Batch Size"] == batch_size]
        
        batch_folder = os.path.join(result_folder, f"batch_size_{batch_size}")
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


def plot_epoch_based_metrics(all_histories, result_folder):
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
        batch_folder = os.path.join(result_folder, f"batch_size_{batch_size}")
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


    
def plot_all_figures(batch_size, epoch, history, y_true_labels, y_pred_labels, y_pred_probs, categories, result_out, model_name):
    plt.figure()
    plt.plot(history.history['accuracy'], linestyle='--')
    plt.plot(history.history['val_accuracy'], linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_accuracy_plot.png'))
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], linestyle='--')
    plt.plot(history.history['val_loss'], linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_loss_plot.png'))
    plt.close()

    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_confusion_matrix_normalized.png'))
    plt.close()

    # 4. ROC Curve Plot for each class in a one-vs-rest fashion
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10', len(categories))  # Get a color map with enough colors
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 'x', 'v', '^', 'd', '*']

    label_encoder = LabelEncoder()
    if isinstance(y_true_labels[0], str) or isinstance(y_true_labels[0], bool):
        y_true_labels = label_encoder.fit_transform(y_true_labels)
    else:
        y_true_labels = np.array(y_true_labels)

    if isinstance(y_pred_labels[0], str) or isinstance(y_pred_labels[0], bool):
        y_pred_labels = label_encoder.transform(y_pred_labels)
    else:
        y_pred_labels = np.array(y_pred_labels)

    if len(categories) == 2:
        # Plotting for the negative class (0)
        y_true_binary = (y_true_labels == 0).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, 0])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors(0), linestyle=line_styles[1], marker=markers[1], label=f'{categories[0]} (AUC = {roc_auc:.4f})')
        
    else:
        for i, class_name in enumerate(categories):
            y_true_binary = (y_true_labels == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors(i), linestyle=line_styles[i % len(line_styles)], marker=markers[i % len(markers)], label=f'{class_name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_roc_curve.png'))
    plt.close()

    # report = classification_report(y_true_labels, y_pred_labels, target_names=categories, output_dict=True)
    report = classification_report(y_true_labels, y_pred_labels, target_names=categories, output_dict=True, zero_division=0)

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
    
    # Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(categories):
        y_true_binary = (y_true_labels == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_probs[:, i])
        plt.plot(recall, precision, label=f'{class_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(result_out, f'{model_name}_bs{batch_size}_ep{epoch}_precision_recall.png'))
    print(f"Precision-recall curve saved to {os.path.join(result_out, f'{model_name}_bs{batch_size}_ep{epoch}_precision_recall.png')}")
    plt.close()


    print(f"All plots saved to {result_out}")

# Load and resize images
def load_and_resize_images(folder, categories, target_size=(224, 224), image_extensions=('.jpg', '.jpeg', '.png')):
    images = []
    labels = []
    for category in categories:
        category_folder = os.path.join(folder, category)
        for filename in os.listdir(category_folder):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                img_path = os.path.join(category_folder, filename)
                label = category
                img = load_img(img_path, target_size=target_size)  # Load and resize the image
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(label)
    return np.array(images), labels

# Create Keras model with pre-trained base models
def create_keras_model(base_model_name, input_shape, num_classes, save_path=None):
    if base_model_name == 'ViT_B16':
        # Initialize ViT_B16 model
        vit_model = vit.vit_b16(
            image_size=224,   # Ensure this matches your input image size
            activation=None,  # No activation function, extracting features
            pretrained=True,  # Use pre-trained weights
            include_top=False,  # Exclude the top classification layer
            pretrained_top=False  # Don't include the pre-trained classification head
        )
        x = vit_model.output
        # No GlobalAveragePooling2D since the output is 2D
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        output = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=vit_model.input, outputs=output)
    else:
        base_model = None  # Initialize the base_model variable
        if base_model_name == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'VGG19':
            base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'ResNet152':
            base_model = ResNet152(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'InceptionV3':
            base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'EfficientNetB7':
            base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'MobileNetV2':
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'Xception':
            base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'DenseNet121':
            base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        else:
            raise ValueError("Invalid model name")

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        output = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Save the model architecture as an image
    if save_path:
        plot_model(model, to_file=os.path.join(save_path, f'{base_model_name}_architecture.png'), show_shapes=True, show_layer_names=True)
        print(f"Model architecture saved to {os.path.join(save_path, f'{base_model_name}_architecture.png')}")

        summary_file = os.path.join(save_path, f'{base_model_name}_summary.txt')
        with open(summary_file, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"Model summary saved to {summary_file}")

    return model


def run_experiment(model_name, epoch_values, batch_size_list, metric_collection, 
                   train_images, train_labels, val_images, val_labels, 
                   test_images, test_labels, result_folder, categories):
    """
    Updated function without EarlyStopping to allow training for all specified epochs.
    """
    performance_metrics = []
    model_histories = []
    all_histories = {}

    # Data augmentation for training
    train_aug = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1./255
    )
    val_aug = ImageDataGenerator(rescale=1./255)
    test_aug = ImageDataGenerator(rescale=1./255)

    # Encoding the labels
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    val_labels_encoded = label_encoder.transform(val_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    num_categories = len(np.unique(train_labels_encoded))

    train_labels_categorical = to_categorical(train_labels_encoded, num_categories)
    val_labels_categorical = to_categorical(val_labels_encoded, num_categories)
    test_labels_categorical = to_categorical(test_labels_encoded, num_categories)

    model_result_out = os.path.join(result_folder, model_name)
    os.makedirs(model_result_out, exist_ok=True)

    if model_name == 'ViT_B16':
        input_shape = (224, 224, 3)
    else:
        input_shape = train_images[0].shape

    model = create_keras_model(model_name, input_shape, num_categories, save_path=model_result_out)

    # Calculate class weights
    class_weight_dict = calculate_class_weights(train_labels_encoded)
    class_weight_save_path = os.path.join(model_result_out, f'{model_name}_class_weights.txt')
    save_class_weights_to_txt(class_weight_dict, class_weight_save_path)

    for batch_size in batch_size_list:
        batch_size_result_out = os.path.join(model_result_out, f'batch_size_{batch_size}')
        os.makedirs(batch_size_result_out, exist_ok=True)

        for epoch in epoch_values:
            epoch_result_out = os.path.join(batch_size_result_out, f'epoch_{epoch}')
            os.makedirs(epoch_result_out, exist_ok=True)

            save_parameters(model_name, batch_size, epoch, model, epoch_result_out)

            # Define callbacks
            best_model_path = os.path.join(epoch_result_out, f'{model_name}_best_model.h5')
            callbacks = [
                ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
            ]

            start_time = datetime.now()

            # Train the model
            history = model.fit(
                train_aug.flow(train_images, train_labels_categorical, batch_size=batch_size),
                epochs=epoch,
                validation_data=val_aug.flow(val_images, val_labels_categorical, batch_size=batch_size),
                callbacks=callbacks,
                verbose=2,
                class_weight=class_weight_dict
            )

            end_time = datetime.now()
            time_taken = end_time - start_time

            # Load the best model from the checkpoint
            best_model = tf.keras.models.load_model(best_model_path)

            # Evaluate on test set
            y_true_labels = label_encoder.inverse_transform(test_labels_encoded)
            y_pred_probs = best_model.predict(test_aug.flow(test_images, batch_size=batch_size, shuffle=False))
            y_pred_labels = label_encoder.inverse_transform(np.argmax(y_pred_probs, axis=1))

            # Calculate test accuracy and other metrics
            test_accuracy = accuracy_score(test_labels, y_pred_labels)
            precision = precision_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
            recall = recall_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
            f1 = f1_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)

            cm = confusion_matrix(y_true_labels, y_pred_labels)
            sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if cm[1, 1] + cm[1, 0] > 0 else 0
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm[0, 0] + cm[0, 1] > 0 else 0

            best_epoch = np.argmax(history.history['val_accuracy']) + 1
            best_val_accuracy = np.max(history.history['val_accuracy'])

            performance_metrics.append({
                "Model": model_name,
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
                "Time Taken": time_taken.total_seconds()
            })
            model_histories.append({
                "batch_size": batch_size,
                "epoch": epoch,
                "history": history.history
            })

            plot_all_figures(batch_size, epoch, history, y_true_labels, y_pred_labels, y_pred_probs, categories, epoch_result_out, model_name)

    performance_df = pd.DataFrame(performance_metrics)
    performance_df.to_csv(os.path.join(result_folder, f'{model_name}_performance_metrics.csv'), index=False)
    print(f"Performance metrics for {model_name} saved to {os.path.join(result_folder, f'{model_name}_performance_metrics.csv')}")

    metric_collection.extend(performance_metrics)

    return {model_name: model_histories}



# Main function
def main():    
    # Directory setup
    base_dir = os.getcwd()
    home_dir = os.path.join(base_dir, 'data2') 
    data_dir = os.path.join(home_dir, 'data2_HAM10000_SPLIT') 
    result_folder = os.path.join(home_dir, 'training_data2_thu_nghiem_1_data2_HAM10000_SPLIT_SOTA_models_v2') 
    os.makedirs(result_folder, exist_ok=True)

    # Define the categories
    categories = ['VASC', 'DF', 'BKL', 'AKIEC', 'BCC', 'NV', 'MEL']
    num_classes = len(categories)

    all_histories = {}
    train_folder = os.path.join(data_dir, 'train')
    val_folder = os.path.join(data_dir, 'val')
    test_folder = os.path.join(data_dir, 'test')

    train_images, train_labels = load_and_resize_images(train_folder, categories)
    val_images, val_labels = load_and_resize_images(val_folder, categories)
    test_images, test_labels = load_and_resize_images(test_folder, categories)

    # batch_size_list = [8, 16, 32, 64]
    # epoch_values = [10, 50, 100, 200]
    
    batch_size_list = [16, 32, 64]
    epoch_values = [200]
    
    # Dictionary để lưu lịch sử huấn luyện của các mô hình
    all_histories = {}
    metric_collection = []
    models = ["ViT_B16", "VGG16", "VGG19", "ResNet152", "InceptionV3", "EfficientNetB0", "EfficientNetB7", "MobileNetV2", "Xception", "DenseNet121"]
    # models = ["VGG16", "VGG19"]
    
    for model_name in models:
        print(f"Running experiment for {model_name}")
        # histories = []
        model_histories = run_experiment(model_name, epoch_values, 
                                         batch_size_list, metric_collection, 
                                         train_images, train_labels, 
                                         val_images, val_labels, 
                                         test_images, test_labels, 
                                         result_folder, categories)
        all_histories.update(model_histories)
        
    metric_collection_df = pd.DataFrame(metric_collection)
    print(metric_collection_df.columns)
    metric_collection_df.to_csv(os.path.join(result_folder, 'overall_performance_metrics.csv'), index=False)

    print(f"Overall performance metrics saved to {os.path.join(result_folder, 'overall_performance_metrics.csv')}")

    # Generate plots using saved histories
    plot_epoch_based_metrics(all_histories, result_folder)
    # Generate combined plots for overall metrics (Precision, Recall, F1-Score, Sensitivity, Specificity)
    plot_combined_metrics(metric_collection, result_folder)  # Call for combined overall metrics

    print("Combined plots saved.")
    
if __name__ == "__main__":
    main()

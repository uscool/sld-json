import os
import numpy as np
import random
import json
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class NoiseGenerator:
    @staticmethod
    def gaussian_noise(image, std_dev=25):
        image_array = np.array(image, dtype=np.float32) / 255.0
        noise = np.random.normal(0, std_dev / 255.0, image_array.shape)
        noisy_image = np.clip(image_array + noise, 0, 1)
        return Image.fromarray((noisy_image * 255).astype(np.uint8))

    @staticmethod
    def salt_and_pepper_noise(image, probability=0.01):
        image_array = np.array(image, dtype=np.float32) / 255.0
        noisy_image = image_array.copy()
        salt_mask = np.random.random(image_array.shape) < probability
        noisy_image[salt_mask] = 1.0
        pepper_mask = np.random.random(image_array.shape) < probability
        noisy_image[pepper_mask] = 0.0

        return Image.fromarray((noisy_image * 255).astype(np.uint8))

    @staticmethod
    def speckle_noise(image, std_dev=0.1):
        image_array = np.array(image, dtype=np.float32) / 255.0
        noise = np.random.normal(1.0, std_dev, image_array.shape)
        noisy_image = np.clip(image_array * noise, 0, 1)
        return Image.fromarray((noisy_image * 255).astype(np.uint8))

    @staticmethod
    def blur_noise(image, blur_radius=1):
        from PIL import ImageFilter

        blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return blurred

    @staticmethod
    def compression_artifacts(image, quality=50):
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    @staticmethod
    def rotation_noise(image, max_angle=5):
        angle = random.uniform(-max_angle, max_angle)
        rotated = image.rotate(angle, expand=False, fillcolor="white")
        return rotated

    @staticmethod
    def brightness_noise(image, factor_range=(0.7, 1.3)):
        from PIL import ImageEnhance

        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def contrast_noise(image, factor_range=(0.7, 1.3)):
        factor = random.uniform(factor_range[0], factor_range[1])
        from PIL import ImageEnhance
        factor = random.uniform(factor_range[0], factor_range[1])
        enhancer = ImageEnhance.Contrast(image)

        return enhancer.enhance(factor)

    @staticmethod
    def combined_noise(image, noise_type="random"):
        noisy_image = image.copy()

        if noise_type == "random":
            noise_types = [
                NoiseGenerator.gaussian_noise,
                NoiseGenerator.salt_and_pepper_noise,
                NoiseGenerator.speckle_noise,
                NoiseGenerator.blur_noise,
            ]
            num_effects = random.randint(1, 2)
            selected_types = random.sample(noise_types, num_effects)

            for noise_func in selected_types:
                try:
                    noisy_image = noise_func(noisy_image)
                except Exception as e:
                    print(f"Warning: {noise_func.__name__} failed: {e}")

        # Always apply brightness/contrast variation
        noisy_image = NoiseGenerator.brightness_noise(noisy_image)
        noisy_image = NoiseGenerator.contrast_noise(noisy_image)

        return noisy_image


class ComponentDataset:
    def __init__(
        self,
        symbol_dir="symbols",
        dataset_dir="dataset",
        noise_types=["gaussian", "salt_pepper", "speckle", "combined"],
    ):
        self.symbol_dir = symbol_dir
        self.dataset_dir = dataset_dir
        self.noise_types = noise_types
        self.noise_gen = NoiseGenerator()

        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "test"), exist_ok=True)

    def generate_dataset(
        self, component_types, samples_per_component=50, train_split=0.8
    ):
        print(
            f"Generating dataset with {samples_per_component} samples per component type..."
        )

        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")

        # Create subdirectories for each component type
        for comp_type in component_types:
            os.makedirs(os.path.join(train_dir, comp_type), exist_ok=True)
            os.makedirs(os.path.join(test_dir, comp_type), exist_ok=True)

        all_labels = []
        all_paths = []

        for comp_type in component_types:
            symbol_path = os.path.join(self.symbol_dir, f"{comp_type}.png")

            if not os.path.exists(symbol_path):
                print(
                    f"Warning: Symbol not found at {symbol_path}. Skipping {comp_type}."
                )
                continue

            original_image = Image.open(symbol_path)
            print(f"\nProcessing {comp_type}...")

            for i in range(samples_per_component):
                # Generate noisy version
                noise_type = random.choice(self.noise_types)

                if noise_type == "gaussian":
                    noisy_img = self.noise_gen.gaussian_noise(
                        original_image, std_dev=random.randint(15, 35)
                    )
                elif noise_type == "salt_pepper":
                    noisy_img = self.noise_gen.salt_and_pepper_noise(
                        original_image, probability=random.uniform(0.005, 0.02)
                    )
                elif noise_type == "speckle":
                    noisy_img = self.noise_gen.speckle_noise(
                        original_image, std_dev=random.uniform(0.05, 0.15)
                    )
                else:  # combined
                    noisy_img = self.noise_gen.combined_noise(original_image)

                # Randomly apply other Effects
                if random.random() < 0.5:
                    noisy_img = self.noise_gen.rotation_noise(noisy_img)
                if random.random() < 0.5:
                    noisy_img = self.noise_gen.blur_noise(
                        noisy_img, blur_radius=random.uniform(0.5, 1.5)
                    )

                # Decide train/test split
                is_train = random.random() < train_split
                split_dir = train_dir if is_train else test_dir

                # Save noisy image
                filename = f"{comp_type}_noisy_{i:04d}_{noise_type}.png"
                filepath = os.path.join(split_dir, comp_type, filename)
                noisy_img.save(filepath)

                all_labels.append(comp_type)
                all_paths.append(filepath)

                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{samples_per_component} samples")

        print(f"\nDataset generation complete!")
        print(f"Total samples: {len(all_paths)}")
        return all_paths, all_labels

    def load_dataset_from_directory(self, split="train"):
        split_dir = os.path.join(self.dataset_dir, split)
        images = []
        labels = []

        for comp_type in os.listdir(split_dir):
            comp_dir = os.path.join(split_dir, comp_type)
            if not os.path.isdir(comp_dir):
                continue

            for image_file in os.listdir(comp_dir):
                image_path = os.path.join(comp_dir, image_file)
                try:
                    img = Image.open(image_path).convert("RGB")
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    labels.append(comp_type)
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")

        return np.array(images), np.array(labels)


class ComponentClassifier:
    def __init__(self, num_classes, input_shape=(100, 100, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._build_model()
        self.history = None
        self.label_encoder = LabelEncoder()

    def _build_model(self):
        model = models.Sequential(
            [
                # First block
                layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    padding="same",
                    input_shape=self.input_shape,
                ),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Second block
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Third block
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Flatten and dense layers
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(128, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        # Encode labels
        unique_labels = np.unique(y_train)
        self.label_encoder.fit(unique_labels)

        y_train_encoded = self.label_encoder.transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)

        # Convert to one-hot
        y_train_onehot = keras.utils.to_categorical(y_train_encoded, self.num_classes)
        y_val_onehot = keras.utils.to_categorical(y_val_encoded, self.num_classes)

        # Training with callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

        print("Training model...")
        self.history = self.model.fit(
            X_train,
            y_train_onehot,
            validation_data=(X_val, y_val_onehot),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        return self.history

    def evaluate(self, X_test, y_test):
        y_test_encoded = self.label_encoder.transform(y_test)
        y_test_onehot = keras.utils.to_categorical(y_test_encoded, self.num_classes)

        loss, accuracy = self.model.evaluate(X_test, y_test_onehot, verbose=0)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Test Loss: {loss:.4f}")

        return accuracy, loss

    def predict(self, image):
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert("RGB")) / 255.0
        else:
            img_array = image

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = self.model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_idx]

        predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]

        return predicted_label, confidence, prediction[0]

    def predict_batch(self, images):
        predictions = self.model.predict(images, verbose=0)
        predicted_indices = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(predicted_indices)

        return predicted_labels, confidences, predictions

    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)
        # rebuild label encoder using component types from generator
        try:
            from components import ComponentGenerator

            gen = ComponentGenerator()
            types = gen.get_component_types()
        except Exception:
            # fallback - if components module not available, leave encoder empty
            types = []
        self.label_encoder = LabelEncoder()
        if types:
            # fit even if empty; fit() requires 1d array
            self.label_encoder.fit(types)
        print(f"Model loaded from {filepath}")
        if types:
            print(f"Label encoder fitted with {len(types)} classes")


class ClassifierPipeline:
    def __init__(
        self,
        symbol_dir="symbols",
        dataset_dir="dataset",
        model_path="component_classifier.h5",
    ):
        self.symbol_dir = symbol_dir
        self.dataset_dir = dataset_dir
        self.model_path = model_path
        self.dataset = None
        self.classifier = None

    def setup_symbols(self):
        if not os.path.exists(self.symbol_dir) or len(os.listdir(self.symbol_dir)) == 0:
            print("Generating component symbols...")
            from components import ComponentGenerator

            gen = ComponentGenerator(self.symbol_dir)
            gen.generate_all_symbols()

    def create_and_train(self, component_types, samples_per_component=50, epochs=50):
        self.setup_symbols()
        self.dataset = ComponentDataset(self.symbol_dir, self.dataset_dir)
        self.dataset.generate_dataset(
            component_types, samples_per_component=samples_per_component
        )
        print("\nLoading training data...")
        X_train, y_train = self.dataset.load_dataset_from_directory("train")
        print(f"Training set: {X_train.shape}")
        print("\nLoading test data...")
        X_test, y_test = self.dataset.load_dataset_from_directory("test")
        print(f"Test set: {X_test.shape}")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        num_classes = len(component_types)
        self.classifier = ComponentClassifier(num_classes=num_classes)
        self.classifier.train(X_train, y_train, X_val, y_val, epochs=epochs)
        self.classifier.evaluate(X_test, y_test)
        self.classifier.save_model(self.model_path)

        return self.classifier

    def load_and_predict(self, image_path, model_path=None):
        if model_path is None:
            model_path = self.model_path
        from components import ComponentGenerator

        gen = ComponentGenerator(self.symbol_dir)
        component_types = gen.get_component_types()

        self.classifier = ComponentClassifier(num_classes=len(component_types))
        self.classifier.load_model(model_path)

        image = Image.open(image_path).convert("RGB")

        label, confidence, all_predictions = self.classifier.predict(image)

        print(f"\nPrediction for {image_path}:")
        print(f"Predicted component: {label}")
        print(f"Confidence: {confidence:.4f}")

        return label, confidence


def visualize_training_history(history, save_path="training_history.png"):
    """Visualize training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.legend()
    ax1.grid()
    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Model Loss")
    ax2.legend()
    ax2.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training history saved to {save_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train classifier for noisy component identification"
    )
    parser.add_argument(
        "--samples", type=int, default=100, help="Samples per component type"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--train", action="store_true", help="Train new model")
    parser.add_argument("--predict", type=str, help="Predict on image file")
    parser.add_argument(
        "--model", type=str, default="component_classifier.h5", help="Model file path"
    )

    args = parser.parse_args()

    pipeline = ClassifierPipeline(model_path=args.model)

    if args.train:
        from components import ComponentGenerator

        gen = ComponentGenerator()
        component_types = gen.get_component_types()

        print(f"Training classifier for {len(component_types)} component types...")
        classifier = pipeline.create_and_train(
            component_types, samples_per_component=args.samples, epochs=args.epochs
        )
        if pipeline.classifier.history:
            visualize_training_history(pipeline.classifier.history)

    if args.predict:
        if not os.path.exists(args.model):
            print(f"Model not found: {args.model}. Train first with --train flag.")
        else:
            pipeline.load_and_predict(args.predict, args.model)


if __name__ == "__main__":
    main()

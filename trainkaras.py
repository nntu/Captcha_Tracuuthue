import tensorflow as tf
import keras
from keras import layers
import numpy as np
from pathlib import Path
import os

class CaptchaModel:
    def __init__(self, img_width=130, img_height=50):
        self.img_width = img_width
        self.img_height = img_height
        self.char_to_num = None
        self.num_to_char = None
        
    @keras.saving.register_keras_serializable()
    class CTCLayer(layers.Layer):
        def __init__(self, char_to_num, name=None, **kwargs):
            super().__init__(name=name, **kwargs)
            self.char_to_num = char_to_num
            
        def call(self, inputs):
            y_true, y_pred = inputs
            batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
            input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
            label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

            input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

            loss = tf.nn.ctc_loss(
                labels=tf.cast(y_true, tf.int32),
                logits=tf.transpose(y_pred, [1, 0, 2]),
                label_length=tf.squeeze(label_length, axis=-1),
                logit_length=tf.squeeze(input_length, axis=-1),
                blank_index=len(self.char_to_num.get_vocabulary())
            )
            self.add_loss(tf.reduce_mean(loss))
            return y_pred

        def compute_output_shape(self, input_shapes):
            return input_shapes[1]

        def get_config(self):
            config = super().get_config()
            config.update({"char_to_num": self.char_to_num})
            return config

    def build_model(self, vocab_size):
        input_img = layers.Input(shape=(self.img_width, self.img_height, 1), name="image", dtype="float32")
        labels = layers.Input(name="label", shape=(None,), dtype="float32")

        # CNN
        x = layers.Conv2D(32, 3, activation="relu", kernel_initializer="he_normal", padding="same")(input_img)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation="relu", kernel_initializer="he_normal", padding="same")(x)
        x = layers.MaxPooling2D()(x)

        # Reshape for RNN
        new_shape = ((self.img_width // 4), (self.img_height // 4) * 64)
        x = layers.Reshape(target_shape=new_shape)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.2)(x)

        # RNN
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

        # Output
        predictions = layers.Dense(vocab_size + 1, activation="softmax")(x)
        
        # CTC layer
        ctc_output = self.CTCLayer(self.char_to_num)([labels, predictions])

        model = keras.Model(inputs=[input_img, labels], outputs=ctc_output)
        model.compile(optimizer=keras.optimizers.Adam())
        return model

    def prepare_dataset(self, data_dir, batch_size=16, train_split=0.9):
        images = sorted(list(map(str, Path(data_dir).glob("*.jpg"))))
        labels = [img.split(os.path.sep)[-1].split(".jpg")[0] for img in images]
        characters = sorted(list(set(char for label in labels for char in label)))
        
        self.char_to_num = layers.StringLookup(vocabulary=characters, mask_token=None)
        self.num_to_char = layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), 
            mask_token=None, 
            invert=True
        )
        
        # Save vocabulary
        with open("vocab.txt", "w") as f:
            for char in self.char_to_num.get_vocabulary():
                f.write(f"{char}\n")
        
        split_idx = int(len(images) * train_split)
        train_images = np.array(images[:split_idx])
        val_images = np.array(images[split_idx:])
        train_labels = np.array(labels[:split_idx])
        val_labels = np.array(labels[split_idx:])
        
        def encode_sample(img_path, label):
            img = tf.io.read_file(img_path)
            img = tf.io.decode_png(img, channels=1)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [self.img_height, self.img_width])
            img = tf.transpose(img, [1, 0, 2])
            label = self.char_to_num(tf.strings.unicode_split(label, "UTF-8"))
            return {"image": img, "label": label}
        
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = (
            train_dataset.map(encode_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_dataset = (
            val_dataset.map(encode_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        return train_dataset, val_dataset, len(characters)

    def train_and_save(self, data_dir, output_path, epochs=100, batch_size=16):
        train_dataset, val_dataset, vocab_size = self.prepare_dataset(data_dir, batch_size)
        model = self.build_model(vocab_size)
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(output_path, save_best_only=True)
        ]
        
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        return model

if __name__ == "__main__":
    captcha_model = CaptchaModel()
    model = captcha_model.train_and_save("./jpg/", "captcha.keras")
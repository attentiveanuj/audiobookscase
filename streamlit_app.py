import streamlit as st
import numpy as np
from sklearn import preprocessing
import tensorflow as tf

def preprocess_data(raw_dataset):
    shuffled_indices = np.arange(raw_dataset.shape[0])
    np.random.shuffle(shuffled_indices)

    raw_shuffled_dataset = raw_dataset[shuffled_indices]

    inputs_prior = raw_dataset[:, 1:-1]
    targets_prior = raw_dataset[:, -1]

    num_of_type_1_targets = int(sum(targets_prior))
    num_of_type_0_targets_counter = 0
    indices_to_be_deleted = []  # to create balance
    for i in range(targets_prior.shape[0]):
        if targets_prior[i] == 0:
            num_of_type_0_targets_counter += 1
            if num_of_type_0_targets_counter > num_of_type_1_targets:
                indices_to_be_deleted.append(i)

    inputs = np.delete(inputs_prior, indices_to_be_deleted, axis=0)
    targets = np.delete(targets_prior, indices_to_be_deleted, axis=0)

    inputs_scaled = preprocessing.scale(inputs)

    shuffled_indices = np.arange(inputs_scaled.shape[0])
    np.random.shuffle(shuffled_indices)

    inputs = inputs_scaled[shuffled_indices]
    targets = targets[shuffled_indices]

    return inputs, targets

def train_model(train_inputs, train_targets, validation_inputs, validation_targets):
    input_size = 10
    hidden_layer_size = 200
    output_size = 2

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
        tf.keras.layers.Dense(hidden_layer_size, activation='tanh'),
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

    max_epochs = 50
    batch_size = 100

    st.write("Epoch\tLoss\tAccuracy")
    for epoch in range(max_epochs):
        model.fit(
            train_inputs,
            train_targets,
            validation_data=(validation_inputs, validation_targets),
            epochs=1,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )

        train_loss, train_accuracy = model.evaluate(train_inputs, train_targets, verbose=0)
        validation_loss, validation_accuracy = model.evaluate(validation_inputs, validation_targets, verbose=0)

        st.write(f"{epoch + 1}\t{train_loss:.4f}\t{train_accuracy * 100:.2f}%\t{validation_accuracy * 100:.2f}%")

    return model

def main():
    st.title("Audiobooks Data Classifier")

    # File Upload
    uploaded_file = st.file_uploader("Upload your dataset", type=['csv'])

    if uploaded_file is not None:
        st.write("File uploaded successfully!")

        # Load and preprocess data
        raw_dataset = np.loadtxt(uploaded_file, delimiter=',')
        inputs, targets = preprocess_data(raw_dataset)

        # Train-validation split
        num_train_samples = int(0.8 * targets.shape[0])
        num_validation_samples = int(0.1 * targets.shape[0])

        train_inputs = inputs[:num_train_samples]
        train_targets = targets[:num_train_samples]
        validation_inputs = inputs[num_train_samples:num_train_samples + num_validation_samples]
        validation_targets = targets[num_train_samples:num_train_samples + num_validation_samples]

        # Train the model
        st.write("Training the model...")
        model = train_model(train_inputs, train_targets, validation_inputs, validation_targets)

        # Display results
        st.write("Training and validation accuracy:")
        train_loss, train_accuracy = model.evaluate(train_inputs, train_targets)
        validation_loss, validation_accuracy = model.evaluate(validation_inputs, validation_targets)
        st.write(f"Training Loss: {train_loss}, Training Accuracy: **{train_accuracy * 100:.2f}%**")
        st.write(f"Validation Loss: {validation_loss}, Validation Accuracy: **{validation_accuracy * 100:.2f}%**")

        # Test the model
        st.write("Testing the model...")
        test_loss, test_accuracy = model.evaluate(inputs[num_train_samples + num_validation_samples:],
                                                  targets[num_train_samples + num_validation_samples:])
        st.write(f"Test Loss: {test_loss}, Test Accuracy: **{test_accuracy * 100:.2f}%**")

if __name__ == "__main__":
    main()

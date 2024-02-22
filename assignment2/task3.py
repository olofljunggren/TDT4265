import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.02
    batch_size = 32
    # neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
    neurons_per_layer = [54, 54, 10]
    momentum_gamma = 0.9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )
    trainer = SoftmaxTrainer(
        momentum_gamma,
        use_momentum,
        model,
        learning_rate,
        batch_size,
        shuffle_data,
        X_train,
        Y_train,
        X_val,
        Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    # Second set hyperparameters
    num_epochs = 50
    learning_rate = 0.02
    batch_size = 32
    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
    momentum_gamma = 0.9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    use_relu = False

    # Train a new model with new parameters
    model_improved = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )
    trainer_improved = SoftmaxTrainer(
        momentum_gamma,
        use_momentum,
        model_improved,
        learning_rate,
        batch_size,
        shuffle_data,
        X_train,
        Y_train,
        X_val,
        Y_val,
    )
    train_history_improved, val_history_improved = trainer_improved.train(num_epochs)

    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"], "Two hidden layers with 54 nodes", npoints_to_average=10)
    utils.plot_loss(
        train_history_improved["loss"],
        "Ten hidden layers with 64 nodes",
        npoints_to_average=10,
    )
    plt.ylabel("Validation loss")
    plt.legend()
    plt.ylim([0, 0.4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 0.95])
    utils.plot_loss(val_history["accuracy"], "Two hidden layers with 54 nodes")
    utils.plot_loss(
        val_history_improved["accuracy"], "Ten hidden layers with 64 nodes"
    )
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    softmax = model.forward(X)
    estimates = np.argmax(softmax,axis=1)
    correct = 0
    total = len(estimates)

    for index, target in enumerate(targets):
        if target[estimates[index]] == 1:
            correct += 1

    accuracy = correct/total
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """

        estimate = self.model.forward(X_batch)
        self.model.backward(X_batch, estimate, Y_batch)
        self.model.w = self.model.w - self.learning_rate*self.model.grad
        loss = cross_entropy_loss(Y_batch, estimate)
        
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # # Intialize model
    # model = SoftmaxModel(l2_reg_lambda)
    # # Train model
    # trainer = SoftmaxTrainer(
    #     model, learning_rate, batch_size, shuffle_dataset,
    #     X_train, Y_train, X_val, Y_val,
    # )
    # train_history, val_history = trainer.train(num_epochs)

    # print("Final Train Cross Entropy Loss:",
    #       cross_entropy_loss(Y_train, model.forward(X_train)))
    # print("Final Validation Cross Entropy Loss:",
    #       cross_entropy_loss(Y_val, model.forward(X_val)))
    # print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    # print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    # plt.ylim([0.2, .6])
    # utils.plot_loss(train_history["loss"],
    #                 "Training Loss", npoints_to_average=10)
    # utils.plot_loss(val_history["loss"], "Validation Loss")
    # plt.legend()
    # plt.xlabel("Number of Training Steps")
    # plt.ylabel("Cross Entropy Loss - Average")
    # plt.savefig("task3b_softmax_train_loss.png")
    # plt.show()

    # # Plot accuracy
    # plt.ylim([0.89, .93])
    # utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    # utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    # plt.xlabel("Number of Training Steps")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.savefig("task3b_softmax_train_accuracy.png")
    # plt.show()

    # # Train a model with L2 regularization (task 4b)
    # model1 = SoftmaxModel(l2_reg_lambda=1.0)
    # trainer = SoftmaxTrainer(
    #     model1, learning_rate, batch_size, shuffle_dataset,
    #     X_train, Y_train, X_val, Y_val,
    # )
    # train_history_reg01, val_history_reg01 = trainer.train(num_epochs)

    # # Plotting of softmax weights (Task 4b)
    # utils.plot_loss(train_history_reg01["loss"],
    #                 "Training Loss", npoints_to_average=10)
    # utils.plot_loss(val_history_reg01["loss"], "Validation Loss")
    # plt.legend()
    # plt.xlabel("Number of Training Steps")
    # plt.ylabel("Cross Entropy Loss - Average")
    # plt.savefig("task4b_softmax_l2_train_loss.png")
    # plt.show()

    # # Organize weights for lambda = 0.0
    # weight0 = np.empty((28,0))
    # for i in range(10):
    #     temp = model.w[:784,i].reshape((28,28))
    #     weight0 = np.block([weight0,temp])
    # plt.imsave("task4b_softmax_weight.png", weight0, cmap="gray")

    # # Organize weights for lambda = 1.0
    # weight1 = np.empty((28,0))
    # for i in range(10):
    #     temp = model1.w[:784,i].reshape((28,28))
    #     weight1 = np.block([weight1,temp])
    # plt.imsave("task4b_softmax_weight_l2.png", weight1, cmap="gray")


    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_norm = []
    val_history_l2 = {}
    l2_lambdas = [1, .1, .01, .001]
    for lambda_val in l2_lambdas:
        model2 = SoftmaxModel(l2_reg_lambda=lambda_val)
        trainer = SoftmaxTrainer(
            model2, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        _, val_history_l2[lambda_val] = trainer.train(num_epochs)
        l2_norm.append(np.linalg.norm(model2.w))

    for lambda_val in l2_lambdas:
        utils.plot_loss(val_history_l2[lambda_val]["loss"], f"Validation Loss {lambda_val}")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task4c_softmax_l2_val_acc.png")
    plt.show()


    # Task 4d - Plotting of the l2 norm for each weight
    plt.plot(l2_lambdas, l2_norm)
    plt.xlabel("Lambda - regularization size")
    plt.ylabel("weight size l2 norm")
    plt.savefig("task4d_l2_reg_norms.png")
    plt.show()


if __name__ == "__main__":
    main()

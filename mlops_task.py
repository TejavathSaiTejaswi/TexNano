from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import optuna
import wandb

wandb.init(project='TexNano', name='wandb_optuna')


# def plot_confusion_matrix(cm, step, name):
#     wandb.log({f'Confusion_Matrix_{name}':  wandb.Image(plt.figure(step))})

def plot_confusion_matrix(cm, step, name):
    plt.figure(figsize=(9, 9))
    plt.imshow(cm, interpolation="nearest", cmap="Pastel1")
    plt.title("Confusion matrix", size=15)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(
        tick_marks,
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        rotation=45,
        size=10,
    )
    plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size=10)
    plt.tight_layout()
    plt.ylabel("Actual label", size=15)
    plt.xlabel("Predicted label", size=15)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(
                str(cm[x][y]),
                xy=(y, x),
                horizontalalignment="center",
                verticalalignment="center",
            )
    wandb.log({f'Confusion_Matrix_{name}':  wandb.Image(plt.figure(step))})


def get_hyper_params_from_optuna(trial):
    penality = trial.suggest_categorical("penality", ["l1", "l2", "elasticnet", "none"])

    if penality == "none":
        solver_choices = ["newton-cg", "lbfgs", "sag", "saga"]
    elif penality == "l1":
        solver = "liblinear"
    elif penality == "l2":
        solver_choices = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    elif penality == "elasticnet":
        solver = "saga"

    if not (penality == "l1" or penality == "elasticnet"):
        solver = trial.suggest_categorical("solver_" + penality, solver_choices)

    C = trial.suggest_float("inverse_of_regularization_strength", 0.1, 1)

    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])

    intercept_scaling = trial.suggest_float("intercept_scaling", 0.1, 1.0)

    if penality == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
    else:
        l1_ratio = None
    return penality, solver, C, fit_intercept, intercept_scaling, l1_ratio


def sanity_checks(digits):
    print("Image Data Shape", digits.data.shape)
    print("Label Data Shape", digits.target.shape)

    plt.figure(figsize=(20, 4))

    for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
        plt.title("Training: %i\n" % label, fontsize=20)
    # plt.show()
    wandb.log({'Image_Data_Shape': digits.data.shape})
    wandb.log({'Label_Data_Shape': digits.target.shape})
    wandb.log({'Sanity_Checks_Plot': wandb.Image(plt)})

# def visualize_test(x_test, y_test, predictions):
#     for image, label, prediction in zip(x_test, y_test, predictions):
#         plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
#         plt.title(f"Label: {label}, Prediction {prediction}")
#         plt.show()
def visualize_test(x_test, y_test, predictions, step):
    for i, (image, label, prediction) in enumerate(zip(x_test, y_test, predictions)):
        if not plt.gcf().get_axes():
            # Handle empty plots
            print(f"Warning: Empty plot at step {step}")
            return
        plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
        plt.title(f"Label: {label}, Prediction {prediction}")
        wandb.log({f'Visualize_Test_Plot_{i}':  wandb.Image(plt.figure(step))})






def evaluate_model(logisticRegr, x_test, y_test, predictions):
    score = logisticRegr.score(x_test, y_test)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    precesion = precision_score(y_test, predictions, average=None)
    recall = recall_score(y_test, predictions, average=None)

    # print("Mean Accuracy:", score)
    # print("Balanced Accuracy:", balanced_accuracy)
    # print("Precesion:", precesion)
    # print("Recall:", recall)
    # return score, balanced_accuracy, precesion, recall

    wandb.log({'Mean_Accuracy': score})
    wandb.log({'Balanced_Accuracy': balanced_accuracy})
    wandb.log({'Precision': precesion})
    wandb.log({'Recall': recall})
    return score, balanced_accuracy, precesion, recall


# def show_confusion_matrix(y_test, predictions):
#     cm = confusion_matrix(y_test, predictions)
#     print(cm)
#     plot_confusion_matrix(cm)

def show_confusion_matrix(y_test, predictions, step):
    cm = confusion_matrix(y_test, predictions)
    wandb.log({'Confusion_Matrix': cm})
    plot_confusion_matrix(cm, step, 'Test')

def objective(trial):

    digits = datasets.load_digits()

    sanity_checks(digits)

    x_train, x_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.25, random_state=0
    )

    (
        penality,
        solver,
        C,
        fit_intercept,
        intercept_scaling,
        l1_ratio,
    ) = get_hyper_params_from_optuna(trial)

    logisticRegr = LogisticRegression(
        penalty=penality,
        C=C,
        fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling,
        solver=solver,
        l1_ratio=l1_ratio,
    )
    logisticRegr.fit(x_train, y_train)
    predictions = logisticRegr.predict(x_test)

    # visualize_test(x_test, y_test, predictions)
    visualize_test(x_test, y_test, predictions, trial.number)
    _, balanced_accuracy, _, _ = evaluate_model(
        logisticRegr, x_test, y_test, predictions
    )

    # show_confusion_matrix(y_test, predictions)
    show_confusion_matrix(y_test, predictions, trial.number)

    return balanced_accuracy

# def main():
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=5)

#     trial = study.best_trial

#     print("Balanced Accuracy: {}".format(trial.value))
#     print("Best hyperparameters: {}".format(trial.params))


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    trial = study.best_trial

    wandb.log({'Best_Balanced_Accuracy': trial.value})
    wandb.log({'Best_Hyperparameters': trial.params})

    
if __name__ == "__main__":
    main()







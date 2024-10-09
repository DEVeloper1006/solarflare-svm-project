#Author: Dev Mody
#Created: October 10th, 2024
#Purpose: Implements an SVM model for Solar Flare Prediction which involves feature selection, data normalization, K-Fold CV and different experiments

#Imported Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

#SolarFlareSVM Class
class SolarFlareSVM:
    
    #Initializes X & Y Dataframes, C, and the SVC model itself
    def __init__(self, X, Y, C=1.0):
        self.X = X
        self.Y = Y
        self.C = C
        self.model = SVC(C=self.C, kernel="rbf", gamma='scale')  # SVC instance from sklearn
        self.X_normalized = None
        self.Y_normalized = None

    #Passes in a list of a desired feature combination and updates the feature matrix
    def feature_creation (self, feature_numbers : list):
        if len(feature_numbers) > 4 or len(feature_numbers) < 1: return
        features = pd.DataFrame()
        for feature_number in feature_numbers:
            if feature_number == 1:
                features = pd.concat([features, self.X.iloc[:, :18]], axis=1)
            elif feature_number == 2:
                features = pd.concat([features, self.X.iloc[:, 18:90]], axis=1)
            elif feature_number == 3:
                features = pd.concat([features, self.X.iloc[:, [90]]], axis=1)
            elif feature_number == 4:
                features = pd.concat([features, self.X.iloc[:, 91:]], axis=1)
            else:
                raise ValueError("Feature Number must be 1, 2, 3 or 4")
        self.X = features 

    # Uses Standard Normalization to preprocess the X feature matrix
    def preprocess(self):
        X_ = self.X.to_numpy()
        means = np.mean(X_, axis=0)
        std_devs = np.std(X_, axis=0)

        # Replace any standard deviations of zero with 1 to avoid division by zero
        std_devs[std_devs == 0] = 1

        self.X_normalized = (X_ - means) / std_devs  # Column-wise normalization
        self.Y_normalized = self.Y.to_numpy()
    
    # Train the SVM using the normalized data
    def train(self):
        self.model.fit(self.X_normalized, self.Y_normalized)
    
    # Predict using the trained SVC model
    def predict(self, X_test):
        return self.model.predict(X_test)

    # Perform k-fold cross-validation to return the average accuracy, a list of confusion matrices for each fold, and a list of TSS scores per fold
    def cross_validation(self, k=10, shuffle=True, random=42):
        folds = KFold(n_splits=k, shuffle=shuffle, random_state=random)
        accuracy = []
        tss = []
        matrices = []

        for train_index, test_index in folds.split(self.X_normalized):
            # Split into train and test sets
            X_train, X_test = self.X_normalized[train_index], self.X_normalized[test_index]
            Y_train, Y_test = self.Y_normalized[train_index], self.Y_normalized[test_index]

            # Train the model
            self.model.fit(X_train, Y_train)

            # Predict on the test set
            Y_predicted = self.model.predict(X_test)

            # Calculate accuracy
            accuracy.append(accuracy_score(Y_test, Y_predicted))
            tss.append(self.tss(Y_test, Y_predicted))

            # Confusion matrix and TSS calculation
            matrices.append(confusion_matrix(Y_test, Y_predicted))

        # Return average accuracy and TSS across all folds
        return np.mean(accuracy), tss, matrices
    
    #Returns the TSS score given predicted Y values and actual Y values
    def tss (self, Y_true, Y_pred):
        # Calculate confusion matrix values
        cm = confusion_matrix(Y_true, Y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Calculate Sensitivity and Specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = fp / (tn + fp) if (tn + fp) > 0 else 0

        # Calculate TSS
        return sensitivity - specificity
  
#Function to determine if a flare class of the {pos,neg}_class.npy files are 1 or -1
def classify_flare(flare_class):
    return 0 if flare_class is None else 1

#Gets Confusion Matrix
def get_confusion_matrix(confusion_matrices, list_of_matrices):
    combined_conf_matrix = np.sum(confusion_matrices, axis=0)
    
    # Plot the combined confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(combined_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    
    # Add title and labels
    plt.title(f'Combined Confusion Matrix of {list_of_matrices}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Display the plot
    plt.tight_layout()
    plt.show()
    
#Plots graph for TSS scores for different feature set combinations
def plot_all_tss_comb(all_tss_scores, all_combinations):
    plt.figure(figsize=(10, 6))
    
    # Loop through each combination and plot its TSS scores on the same graph
    for idx, combination in enumerate(all_combinations):
        plt.plot(range(1, 11), all_tss_scores[idx], marker='o', linestyle='-', label=f"Combination {combination}")
    
    # Add title and labels
    plt.title("TSS Scores for All Combinations")
    plt.xlabel("Fold Number")
    plt.ylabel("TSS Score")
    
    # Set ticks and limits for the y-axis
    plt.xticks(range(1, 11))
    plt.ylim(0, 1)
    
    # Add grid and horizontal line at y=0
    plt.grid(True)
    plt.axhline(0, color='red', linestyle='--')
    
    # Add a legend to distinguish between different combinations
    plt.legend(title="Combinations")
    
    # Display the plot
    plt.tight_layout()
    plt.show()
    
#Plots graph for TSS scores for different datasets
def plot_all_tss_dir(all_tss_scores, all_directories):
    plt.figure(figsize=(10, 6))
    
    # Loop through each combination and plot its TSS scores on the same graph
    for idx, directory in enumerate(all_directories):
        plt.plot(range(1, 11), all_tss_scores[idx], marker='o', linestyle='-', label=f"Directory {directory}")
    
    # Add title and labels
    plt.title("TSS Scores for All Directories")
    plt.xlabel("Fold Number")
    plt.ylabel("TSS Score")
    
    # Set ticks and limits for the y-axis
    plt.xticks(range(1, 11))
    plt.ylim(0, 1)
    
    # Add grid and horizontal line at y=0
    plt.grid(True)
    plt.axhline(0, color='red', linestyle='--')
    
    # Add a legend to distinguish between different combinations
    plt.legend(title="Directories")
    
    # Display the plot
    plt.tight_layout()
    plt.show()
    
#Uses itertools to return a list representing the powerset of a set
def power_set (s):
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1)))

#Returns the default X and Y dataframes
def get_X_Y (directory : str, shuffle=True):
    fs_ii_pos = np.load(f"{directory}/pos_features_main_timechange.npy", allow_pickle=True)
    fs_ii_neg = np.load(f"{directory}/neg_features_main_timechange.npy", allow_pickle=True)

    fs_iii_pos = np.load(f"{directory}/pos_features_historical.npy", allow_pickle=True)
    fs_iii_neg = np.load(f"{directory}/neg_features_historical.npy", allow_pickle=True)

    fs_iiii_pos = np.load(f"{directory}/pos_features_maxmin.npy", allow_pickle=True)
    fs_iiii_neg = np.load(f"{directory}/neg_features_maxmin.npy", allow_pickle=True)

    labels_pos = np.load(f"{directory}/pos_class.npy", allow_pickle=True)
    labels_neg = np.load(f"{directory}/neg_class.npy", allow_pickle=True)

    # Concatenate the positive and negative features
    fs_pos = np.column_stack((fs_ii_pos, fs_iii_pos, fs_iiii_pos))
    fs_neg = np.column_stack((fs_ii_neg, fs_iii_neg, fs_iiii_neg))

    # Create the Positive DataFrame
    df_pos = pd.DataFrame(fs_pos, columns=[f'FS_Feature_{i + 1}' for i in range(fs_pos.shape[1])])
    df_pos['FLARE'] = [classify_flare(label[2]) for label in labels_pos]

    # Create the Negative DataFrame
    df_neg = pd.DataFrame(fs_neg, columns=[f'FS_Feature_{i + 1}' for i in range(fs_neg.shape[1])])
    df_neg['FLARE'] = [classify_flare(label[2]) for label in labels_neg]

    # Combine the data
    df_combined = pd.concat([df_pos, df_neg], axis=0).reset_index(drop=True)
    
    if not shuffle:
        data_order = np.load(f"{directory}/data_order.npy", allow_pickle=True)
        df_combined = df_combined.iloc[data_order]

    X = df_combined.iloc[:, :-1]
    Y = df_combined.iloc[:, -1]
    return X, Y

#Feature Experiment
def feature_experiment ():
    X, Y = get_X_Y("./data/data-2010-15")
    all_combinations = [list(tup) for tup in power_set([1,2,3,4])[1:]]
    all_tss_scores = []
    best_tss = 0
    best_combination = []
    for combination in all_combinations:
        my_svm = SolarFlareSVM(X, Y, C=1)
        my_svm.feature_creation(combination)
        my_svm.preprocess()
        accuracy, tss_scores, all_matrices = my_svm.cross_validation(k=10)
        avg_tss = np.mean(tss_scores)
        if avg_tss > best_tss:
            best_combination = combination
            best_tss = avg_tss
        all_tss_scores.append(tss_scores)
        get_confusion_matrix(all_matrices, combination)
        print(f"Combination {combination}:\n\tAverage Accuracy: {accuracy}\n\tAverage TSS Score: {avg_tss}\n")
    plot_all_tss_comb(all_tss_scores, all_combinations)
    print(f"Best Combination: {best_combination}\n")
    return best_combination

#Data Experiment
def data_experiment (best_combination : list):
    directories = ['./data/data-2010-15', './data/data-2020-24']
    all_tss_scores = []
    for directory in directories:
        X, Y = get_X_Y(directory)
        my_svm = SolarFlareSVM(X, Y, C=1)
        my_svm.feature_creation(best_combination)
        my_svm.preprocess()
        accuracy, tss_scores, all_matrices = my_svm.cross_validation(k=10)
        all_tss_scores.append(tss_scores)
        get_confusion_matrix(all_matrices, directory)
        print(f"Dataset {directory}:\n\tAverage Accuracy: {accuracy}\n\tFeature Combination: {best_combination}\n\tAverage TSS Score: {np.mean(tss_scores)}\n")
    plot_all_tss_dir(all_tss_scores, directories)
    
# No Shuffle Experiment (Follows the order from data_order.npy and disables shuffling in Cross Validation for KFold)
def no_shuffle_experiment(best_combination : list):

    X, Y = get_X_Y("./data/data-2010-15", shuffle=False)
    my_svm = SolarFlareSVM(X, Y, C=1)
    my_svm.feature_creation(best_combination)
    my_svm.preprocess()

    accuracy, tss_scores, all_matrices = my_svm.cross_validation(k=10, shuffle=False, random=None)
    get_confusion_matrix(all_matrices, "No Shuffle Experiment")
    print(f"Accuracy: {accuracy}, TSS: {np.mean(tss_scores)}")
    
#Model to be trained
X, Y = get_X_Y("./data/data-2010-15")
my_svm = SolarFlareSVM(X, Y, C=1.821)
my_svm.feature_creation([1,4])
my_svm.preprocess()

#Gets the normalized X values for the Testing
X2, Y2 = get_X_Y("./data/data-2020-24")
my_svm2 = SolarFlareSVM(X2, Y2, C=1.821)
my_svm2.feature_creation([1,4])
my_svm2.preprocess()

my_svm.train() #Trains the Model
Y_predict = my_svm.predict(my_svm2.X_normalized) #Getting predicts for the model from the Testing Set
Y_predict2 = my_svm.predict(my_svm.X_normalized) #Gets predics for the model from the Training Set
print(accuracy_score(my_svm.Y_normalized, Y_predict2), my_svm.tss(my_svm.Y_normalized, Y_predict2)) #Accuracy of Training
print(accuracy_score(my_svm2.Y_normalized, Y_predict), my_svm.tss(my_svm2.Y_normalized, Y_predict)) #Accuracy of Testing

#Calls each experiment
best_combination = feature_experiment()    
data_experiment(best_combination=best_combination)
no_shuffle_experiment(best_combination=best_combination)

import pandas as pd
import numpy as np
import math
import time
from collections import namedtuple
import matplotlib.pyplot as plt

#load dataset with pd.csv, and apply column names
def load_data(path, column_names):
    data = pd.read_csv(path, header = None)
    data.columns = column_names
    return data

#calculates time it takes to run an algorithm on a dataset
def time_func(name, func):
  start = time.time()
  result = func()
  end = time.time()
  took_seconds = int(end - start)
  print(f"{name} executed in {took_seconds}s")
  return result

def fill_missing_with_mean_in_column(input_df, column): #applies mean to missing data only for one column
    result = input_df.copy()
    #coverting data into numberic in order to take a mean; error="coerce" forces the data into numberic form.
    result[column] = pd.to_numeric(result[column], errors='coerce')
    result[column].fillna(value=result[column].mean(), inplace=True) #fillna applies the mean of a column to cells with NA values
    return result

#applies mean to missing data for multiple columns in a dataframe by looping over columns
def fill_missing_with_mean_all_columns(input_df):
    result = input_df.copy()
    for column in result.columns:
        result = fill_missing_with_mean_in_column(result, column)
    return result

def fill_missing_with_mode_in_column(input_df, column, downcast=None):
    result = input_df.copy()
    #coverting data into numberic in order to take a mean
    result[column] = pd.to_numeric(result[column], errors='coerce', downcast=downcast)
    result[column].fillna(value=result[column].mode()[0], inplace=True) #fillna applies the mean of a column to cells with NA values
    return result

# apply mode of each column to fill the NA in that column using the above fill_missing_with_mode_in_column column
def fill_missing_with_mode_all_columns(input_df):
    result = input_df.copy()
    for column in result.columns:
        result = fill_missing_with_mode_in_column(result, column)
    return result

def ordinal_coding(input_df, column, ordered_catagories):
    distinct_column_values = list(input_df[column].unique())
    for value in distinct_column_values:
        if not value in ordered_catagories:#raise exception if the values in the ordered_category are not found in the column
            raise Exception(f'No order defined for {value}')
    #takes in ordered_categories to produce dictionary with values starting at 1 in an increasing order
    other_dict = {cat: idx for (cat, idx) in zip(ordered_catagories, range(1,len(ordered_catagories)+1))}
    replace_dict = dict()
    replace_dict[column] = other_dict
    #replaces values in a column with ordinal encoding
    return input_df.replace(replace_dict)
    
def nominal_categorical_to_one_hot(input_df, column):
    #produces columns with values 0 or 1 for each catogory in a feature
    dummies = pd.get_dummies(input_df[column], prefix=column)
    #new rows are concated to the existing table
    result = pd.concat([input_df, dummies], axis=1)
    del result[column]
    return result

def discretize(input_df, column, number_of_bins, equal_width=False):
    result = input_df.copy()
    labels=range(1, number_of_bins + 1)
    if equal_width: # data points are divided into equal sized bins
        discretized_column = pd.cut(result[column], bins=number_of_bins, labels=labels)
    else: #bins are divided into equal number of data points
        discretized_column = pd.qcut(result[column], q=number_of_bins, labels=labels)
    result[column] = discretized_column
    return result

#standardize one column
def standardize_column(input_df, column): 
    result = input_df.copy()
    series = input_df[column]
    #z-score stadardization
    result[column]=(series-series.mean())/series.std()
    return result

def standardize_columns(df, columns):
  #standardize all columns
    for column in columns:
        df = standardize_column(df, column)
    return df

def load_abalone_data():
    abalone_column_names = ["Sex", "Length", "Diameter", "Height", "Whole Weight", 
                            "Shucked Weight", "Viscera Weight", "Shell Weight", "Rings"]
    abalone_data = load_data("/content/drive/MyDrive/Colab Notebooks/abalone.data",abalone_column_names )

    #sex category to one hot
    abalone_data = nominal_categorical_to_one_hot(abalone_data, "Sex")

    #standardize numerical columns
    abalone_data = standardize_columns(abalone_data, ["Length", "Diameter", "Height", "Whole Weight",
                                                      "Shucked Weight", "Viscera Weight", "Shell Weight"])
    return abalone_data

def load_house_votes_data():
    #loading house votes data and applying column names
    house_votes_data_columns = ["Class Name", "Handicapped-infants", "water-project-cost-sharing",
                                "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid",
                                "religious-groups-in-schools", "anti-satellite-test-ban","aid-to-nicaraguan-contras", 
                                "mx-missile", "immigration","synfuels-corporation-cutback","education-spending",
                                "superfund-right-to-sue", "crime", "duty-free-exports","export-administration-act-south-africa"]
    house_votes_data = load_data ("/content/drive/MyDrive/Colab Notebooks/house-votes-84.data",house_votes_data_columns )
    house_votes_cat = ["y", "n", "?"]
    #applying one-hot to house votes columns 
    for column in ["Handicapped-infants", "water-project-cost-sharing",
                                "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid",
                                "religious-groups-in-schools", "anti-satellite-test-ban","aid-to-nicaraguan-contras", 
                                "mx-missile", "immigration","synfuels-corporation-cutback","education-spending",
                                "superfund-right-to-sue", "crime", "duty-free-exports","export-administration-act-south-africa"]:
        house_votes_data = ordinal_coding(house_votes_data, column, house_votes_cat)
        #house_votes_data = nominal_categorical_to_one_hot(house_votes_data, column)
    
    return house_votes_data
     
def load_car_data():
    #loading car data and applying column names
    car_data_columns  = ["Buying", "Maint", "Doors", "Persons", "Lug_boot", "Safety"]
    car_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/car.data")
    # dropped the last column as the information for the attribute was missing in .NAMES file
    car_data = car_data.drop(car_data.columns[[6]], axis=1)
    car_data.columns = car_data_columns

    #adding ordinal categories to a list
    lug_boot_cat = ["small", "med", "big"]
    safety_cat = ["low", "med", "high"]
    buying_cat = ['vhigh', 'high', 'med', 'low']
    maint_cat = ['vhigh', 'high', 'med', 'low']

    #Applying ordinal coding to multiple columns using a list of ordered categories for each column
    car_data = ordinal_coding(car_data, "Lug_boot", lug_boot_cat)
    car_data = ordinal_coding(car_data, "Safety", safety_cat)
    car_data = ordinal_coding(car_data, "Buying", buying_cat)
    car_data = ordinal_coding(car_data, "Maint", maint_cat)

    #replace more and 5more in the persons and doors column with integer 5
    car_data["Persons"]= car_data["Persons"].replace({"more": 5}).astype("int64")
    car_data["Doors"] = car_data["Doors"].replace({"5more": 5}).astype("int64")
   #subtracting one so ordinal coding starts from one for hot encoding in logistic regression
    car_data["Safety"] = car_data["Safety"] - 1
    return car_data

def load_machine_data():
    machine_data_columns = ["Vendor Name", "Model Name","MYCT","MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    machine_data = load_data ("/content/drive/MyDrive/Colab Notebooks/machine.data",machine_data_columns )

    #one-hot encoding of vendor name
    #machine_data = nominal_categorical_to_one_hot(machine_data, "Vendor Name")
    machine_data = machine_data.drop("Vendor Name", axis = 1)
    #dopping model name as it is defined as non-predictive in the NAME file
    machine_data = machine_data.drop("Model Name", axis = 1)
    
    #standardize numerical columns
    machine_data = standardize_columns(machine_data, ["MYCT","MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP"])
    
    return machine_data


def load_forest_fires_data():
    forest_fires_data_columns = ["X", "Y", "Month", "Day", "FFMC", "DMC", "DC", "ISI", "temp", 
                                "RH", "Wind", "Rain", "Area"]
    forest_fires_data = load_data("/content/drive/MyDrive/Colab Notebooks/forestfires.csv", forest_fires_data_columns).drop([0])#dropping row 0 to remove column names from the first row
    # forest_fires_data = nominal_categorical_to_one_hot(forest_fires_data, "Month")
    # forest_fires_data = nominal_categorical_to_one_hot(forest_fires_data, "Day")
    
    month_cat = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    day_cat = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

    #ordinal coding the values with string categoris to numerical categories
    forest_fires_data = ordinal_coding(forest_fires_data, "Month", month_cat)
    forest_fires_data = ordinal_coding(forest_fires_data, "Day", day_cat)
    #converting the columns to float
    forest_fires_data = fill_missing_with_mean_all_columns(forest_fires_data)
    for column in ["X", "Y", "Month", "Day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "Wind", "Rain", "Area"]:
        forest_fires_data[column] = forest_fires_data[column].astype("float64")
    
    #standardization of columns
    forest_fires_data = standardize_columns(forest_fires_data, ["X", "Y", "Month", "Day", "FFMC", 
                                                                "DMC", "DC", "ISI", "temp", 
                                                                "RH", "Wind", "Rain"])
    return forest_fires_data

def load_breast_cancer_data():
  # loading breast cancer data and adding column names
    breast_cancer_wisc_data_column_names = ["Sample code number","Clump Thickness","Uniformity of Cell Size", "Uniformity of Cell Shape",
                                       "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli",
                                       "Mitoses", "Class"]
    breast_cancer_wisc_data = load_data("/content/drive/MyDrive/Colab Notebooks/breast-cancer-wisconsin.data", breast_cancer_wisc_data_column_names)
    breast_cancer_wisc_data = fill_missing_with_mode_in_column(breast_cancer_wisc_data, column="Bare Nuclei", downcast='integer')

    #breast_cancer_wisc_data= breast_cancer_wisc_data.astype("int64", errors = "ignore")
    breast_cancer_wisc_data["Class"] = breast_cancer_wisc_data["Class"].replace({2: 0}).astype("int64")
    breast_cancer_wisc_data["Class"] = breast_cancer_wisc_data["Class"].replace({4: 1}).astype("int64")

    breast_cancer_wisc_data = standardize_columns(breast_cancer_wisc_data, ["Clump Thickness","Uniformity of Cell Size", "Uniformity of Cell Shape",
                                       "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli",
                                       "Mitoses" ])
    breast_cancer_wisc_data = breast_cancer_wisc_data.drop("Sample code number", axis = 1)
    return breast_cancer_wisc_data



TrainAndTestSet = namedtuple('TrainAndTestSet', "train test")
TrainAndTestAndValidationSet = namedtuple('TrainAndTestAndValidationSet', "train test validation")

SAMPLE_FRAC=0.2
SAMPLE_SEED = 29573
def split_with_stratify(input_df, num_folds, column):
    # group the dataframe by the column
    grouped_by_column_values = input_df.groupby(column)
    # for each grouped value, divide that into number of folds
    each_group_split_into_folds = [np.array_split(group, num_folds) for _, group in grouped_by_column_values]
    
    result = []
    for fold_number in range(0, num_folds):
        # for the first fold, take the [0] from each group, etc
        arranged_by_folds = [group[fold_number] for group in each_group_split_into_folds]
        # suffle each split because they were semi-sorted by the grouping
        result.append(pd.concat(arranged_by_folds).sample(frac=SAMPLE_FRAC,random_state = SAMPLE_SEED))

    return result

def split_into_k_folds(input_df, num_folds, stratify_column=None, extract_validation_set=False):
    result = []

    if stratify_column:
        split = split_with_stratify(input_df, num_folds, stratify_column)
    else:
        # suffle the dataset and divide into num_folds dataframes of nearly equal size
        split = np.array_split(input_df.sample(frac=SAMPLE_FRAC, random_state = SAMPLE_SEED), num_folds)
    for fold_number in range(0, num_folds):
        #creating train and test sets
        test_set = split[fold_number]
        train_folds = split[:fold_number] + split[fold_number:]
        if extract_validation_set:
            # if we want a validation set, take out the first fold from the training set and use it for validation
            result.append(TrainAndTestAndValidationSet(
                pd.concat(train_folds[:1]).reset_index(drop=True),
                test_set.reset_index(drop=True),
                train_folds[0].reset_index(drop=True)))
        else: 
            result.append(TrainAndTestSet(pd.concat(train_folds).reset_index(drop=True), test_set.reset_index(drop=True)))
    
    return result



#converts np array from 1d to 2d
def np_1d_to_2d(arr):
  return arr.reshape(arr.shape[0],-1)


FLAG_SHOULD_PRINT_WEIGHT_UPDATE = False
FLAG_SHOULD_PRINT_ACTIVATIONS = False
FLAG_SHOULD_PRINT_GRADIENTS = False

##Code for Logistic regression
##Uses softmax function to make predictions

def one_hot(y, num_classes):
    """
    Computes softmax on all rows

    :param y: zero indexed class labels
    :param num_classes: total number of classes
    :return: one-hot encoded class labels
    """
    
    # same number of rows, one column per class
    y_hot = np.zeros((len(y), num_classes))
    
    # each row gets a one where the preivous zero-index class label is
    y_hot[np.arange(len(y)), y] = 1
    
    return y_hot

def softmax(z):
    """
    Computes softmax on all rows

    :param z: rows of weighted outputs for each class
    :return: rows normalized to sum=1
    """
    
    # for numerical stability
    exp = np.exp(z - np.max(z))
    
    # divide each row by its sum
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])
        
    return exp

def log_reg_fit(X, y, learning_rate, num_iterations, print_loss = False):
    """
    fits weights + bias using gradient descent for multi-class logistic regression

    :param X: training set
    :param y: training labels
    :param learning_rate: learning rate
    :param num_iterations: num iterations
    :return: three-tuple of weights, biases, and loss history
    """
        
    # m is number of training examples
    # n is number of features 
    m, n = X.shape

    # comute total number of classes based on unique counts
    num_classes = len(np.unique(y))
    
    # randomly initialize weights and bias
    w = np.random.random((n, num_classes))
    b = np.random.random(num_classes)
    #print("Randomly initialized weights", w )
    # track losses over iterations
    losses = []

    # encode y into one-hot
    y_hot = one_hot(y, num_classes)
    
    # gradient descent
    for iteration in range(num_iterations):
        
        # Calculating hypothesis/prediction.
        z = X@w + b
        y_hat = softmax(z)
        
        # Calculating the gradient of loss w.r.t w and b.
        w_grad = (1/m)*np.dot(X.T, (y_hat - y_hot)) 
        b_grad = (1/m)*np.sum(y_hat - y_hot)

        if iteration == 1000 and FLAG_SHOULD_PRINT_WEIGHT_UPDATE:
          print("*** START Demonstrating Logistic Regression Weight Updates ***\n")
          print ("After 1000 iterations")
          print("Current Weight", w)
          print("weight gradient", w_grad)
          print("weight updated with learning rate", w - learning_rate * w_grad)
          print("\n*** END Demonstrating Logistic Regression Weight Updates ***\n")
        
        # Updating the parameters.
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad
        
        

        # Printing out the loss at every 500th iteration.
        if print_loss and iteration % 500 == 0:
          loss = -np.mean(np.log(y_hat[np.arange(len(y)), y]))
          losses.append(loss)
          # Calculating loss and appending it in the list.
          # print('Iteration {iteration}==> Loss = {loss}'
                # .format(iteration=iteration, loss=loss))
    
    return w, b, losses

def log_reg_compute_predictions(X, w, b):
    """
    Predict class labels for examples using weights and biases

    :param X: Examples
    :param w: weights
    :param b: bias intercepts
    :return: array of predictions
    """
    
    z = X@w + b
    y_hat = softmax(z)
    
    # Returning the class with highest probability.
    return np.argmax(y_hat, axis=1)

def log_reg_compute_accuracy(y, y_hat):
    """
    comutes accuracy from predictions made for multi-class logistic regression

    :param y: truth labels
    :param y_hat: label predictions
    :return: accuracy between 0 and 1
    """
    return np.sum(y==y_hat)/len(y)

#Code for Linear Regression
#Uses X@W + b to calculate output

def fit_linear(X, y, learning_rate, num_iterations):
    # shape of X: (number of training examples: m, number of    
    # features: n)
    m, n = X.shape    

    # Initializing weights as a matrix of zeros of size: (number
    # of features: n, 1) and bias as 0
    weights = np.zeros((n,1))
    bias = 0
    
    # reshaping y as (m,1) in case your dataset initialized as 
    # (m,) which can cause problems
    y = y.reshape(m,1)
    
    # empty lsit to store losses so we can plot them later 
    # against epochs
    losses = []
    
    # Gradient Descent loop/ Training loop
    for iteration in range(num_iterations):
        # Calculating prediction: y_hat or h(x)
        y_hat = np.dot(X, weights) + bias
       

        if iteration % 500  == 0:
            # Calculting loss
            loss = np.mean((y_hat - y)**2)

        # Appending loss in list: losses
            losses.append(loss)
            print('Iteration {iteration}==> Loss = {loss}'
                .format(iteration=iteration, loss=loss))

        # Calculating derivatives of parameters(weights, and 
        # bias) 
        dw = (1/m)*np.dot(X.T, (y_hat - y))
        db = (1/m)*np.sum((y_hat - y))
      # Updating the parameters: parameter := parameter - lr*derivative
      # of loss/cost w.r.t parameter)
        weights -= learning_rate*dw
        bias -= learning_rate*db
    
    # returning the parameter so we can look at them later
    return weights, bias, losses

# Predicting(calculating y_hat with our updated weights) for the 
# testing/validation     
def predict_lin(X, w, b):
    predicted = np.dot(X, w) + b
    return predicted

def lin_mse(y_hat, y_actual):
    mse = np.mean((y_hat - np_1d_to_2d(y_actual)**2))
    return mse


np.random.seed(0)

class NeuralNet(object):
  pass

class NeuralNetClassification(NeuralNet):
  def __init__(self, nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim):
    """
    Initialize weights & biases to random values

    :param nn_input_dim: number of input features
    :param nn_hdim1: dimensions of hidden layer 1
    :param nn_hdim2: dimensions of hidden layer 2
    :param nn_output_dim: number of classes
    :return: initialized network
    """
    #weights are initialized with really small numbers by diving by the sq root of the input dimentions
    #bias is initilized with zeros with dimentions of the respective hidden layer
    W1 = np.random.randn(nn_input_dim, nn_hdim1) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim1))
    W2 = np.random.randn(nn_hdim1, nn_hdim2) / np.sqrt(nn_hdim1)
    b2 = np.zeros((1, nn_hdim2))
    W3 = np.random.randn(nn_hdim2, nn_output_dim) / np.sqrt(nn_hdim2)
    b3 = np.zeros((1, nn_output_dim))

    self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

  def fit(self, X, y, num_iterations=12500, learning_rate=0.01, regularization=0.01, print_loss=False):
    """
    Fits the model from labeled data

    :param X: input data
    :param y: labels
    :param num_iterations: number of iterations
    :param learning_rate: learning rate
    :param regularization: regularization
    :param print_loss: whether to compute and print loss
    :return: model and losses of iterations
    """
    num_examples = len(X)
    print("shape of the input data", X.shape)
    W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']

    losses = []
    # Gradient descent.
    # hyperbolic function is used as an activation function
    for iteration in range(0, num_iterations):

        # Forward propagation
        z1 = X.dot(W1) + b1
        
        #hyperbolic activation function for layer 1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        #hyperbolic activation function for layer 1
        a2 = np.tanh(z2)

        z3 = a2.dot(W3) + b3
        #final output with softmax function
        a3 = self.softmax(z3)

        # Backpropagation
        #calculating gradients for dA dw and db from output layer
        dA3 = a3
        dA3[range(num_examples), y] -= 1
        dW3 = (a2.T).dot(dA3)
        db3 = np.sum(dA3, axis=0, keepdims=True)

        #calculating gradients for dA, dw and db from hidden layer 2
        dA2 = dA3.dot(W3.T) * (1 - np.power(a2, 2))
        dW2 = np.dot(a1.T, dA2)
        db2 = np.sum(dA2, axis=0, keepdims=True)
        
        #calculating gradients for dA, dw and db from hidden layer 1
        dA1 = dA2.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, dA1)
        db1 = np.sum(dA1, axis=0)

        if iteration == 1000 and FLAG_SHOULD_PRINT_ACTIVATIONS:
            print("*** START Demonstrating Classification Neural Network Activation ***\n")
            print("z1 = X@W1 + b1, at hidden layer 1", z1[0])
            print("tanh(z1), Activation (a1) at hidden layer 1 for classification\n","a1 shape", a1.shape, a1[0])
            print("z2 = a1@W2 + b2, at hidden layer 2", z2[0])
            print("Activation, a2 = tanh(z2) at hidden layer 2 for classification\n","a2 shape", a2.shape, a2[0])
            print("z3 = a2@W3 + b3,  at output layer", z3[0])
            print("final output a3 = softmax (z3) (input x num classes)\n","a1 shape", a3.shape, self.softmax(z3)[0])
            print("\n*** END Demonstrating Classification Neural Network Activation ***\n")

        if iteration == 1000 and FLAG_SHOULD_PRINT_WEIGHT_UPDATE:
            print("*** START Demonstrating Classification Neural Net Weight Updates ***\n")
            print("Starting weight W1", W1[0])
            print("weight W1 gradient", dW1[0])
            print("weight updated with learning rate", (W1 - learning_rate * dW1)[0])
            print("Starting weight W2", W2[0])
            print("weight W2 gradient", dW2[0])
            print("weight updated with learning rate", (W2 - learning_rate * dW2)[0])
            print("Starting weight W3", W3[0])
            print("weight W3 gradient", dW3[0])
            print("weight updated with learning rate", (W3 - learning_rate * dW3)[0])
            print("\n*** END Demonstrating Classification Neural Net Weight Updates ***\n")
          

        # regularization for the weight gradients
        dW3 += regularization * W3
        dW2 += regularization * W2
        dW1 += regularization * W1

        # Update weights & biases from gradient from each layer by multiplying with learning rate 
        W1 += -learning_rate * dW1
        b1 += -learning_rate * db1
        W2 += -learning_rate * dW2
        b2 += -learning_rate * db2
        W3 += -learning_rate * dW3
        b3 += -learning_rate * db3
        
        # Assign new parameters to the model
        self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
        
        # compute loss using change in activation using cross entropy
        
        
        if print_loss and iteration % 500 == 0:
          loss = self.loss(X, y)
          losses.append(loss)
          print("Loss after iteration %i: %f" %(iteration, loss))
    
    return (self.model, losses)

  def loss(self, X, y):
    """
    Computes loss

    :param X: input data
    :param y: labels
    :return: cross entropy loss
    """
    W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
    # Forward propagation to calculate our predictions
    
    z1 = X.dot(W1) + b1
    #hyperbolic activation function
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)
    z3 = a2.dot(W3) + b3
    #final output is computed using softmax function
    a3 = self.softmax(z3)
    # Calculating the loss using cross entropy loss function
    correct_logprobs = -np.log(a3[range(len(X)), y])
    data_loss = np.sum(correct_logprobs)
    return 1./len(X) * data_loss

  def predict(self, X):
    """
    Generates predictions from the model

    :param X: input data
    """
    W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
    # Forward propagation
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)
    z3 = a2.dot(W3) + b3
     #final output is computed using softmax function
    a3 = self.softmax(z3)

    return np.argmax(a3, axis=1)

  def score(self, X, y):
    """
    Scores the model using input data

    :param X: input data
    :param y: labels
    """
    result = self.predict(X)
    score = np.sum(y==result)/len(y)
    return score
    
  def softmax(self, z):
    """
    Computes softmax for a row

    :param z: row to normalize
    """
    exp = np.exp(z)
    a = exp / np.sum(exp, axis=1, keepdims=True)
    return a

class NeuralNetRegression(NeuralNet):
  def __init__(self, nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim):
    """
    Initialize weights & biases to random values

    :param nn_input_dim: number of input features
    :param nn_hdim1: dimensions of hidden layer 1
    :param nn_hdim2: dimensions of hidden layer 2
    :param nn_output_dim: number of classes
    :return: initialized network
    """
    #weights are initialized with really small numbers by diving by the sq root of the input dimentions
    #bias is initilized with zeros with dimentions of the respective hidden layer
    W1 = np.random.randn(nn_input_dim, nn_hdim1) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim1))
    W2 = np.random.randn(nn_hdim1, nn_hdim2) / np.sqrt(nn_hdim1)
    b2 = np.zeros((1, nn_hdim2))
    W3 = np.random.randn(nn_hdim2, nn_output_dim) / np.sqrt(nn_hdim2)
    b3 = np.zeros((1, nn_output_dim))

    self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

  def fit(self, X, y, num_iterations, learning_rate=0.05, regularization=0.01, print_loss=False):
    """
    Fits the model from labeled data

    :param X: input data
    :param y: labels
    :param num_iterations: number of iterations
    :param learning_rate: learning rate
    :param regularization: regularization
    :param print_loss: whether to compute and print loss
    :return: model and losses of iterations
    """
    num_examples = len(X)
    print("shape of the input data", X.shape)
    W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']

    losses = []
    # Gradient descent.
    for iteration in range(0, num_iterations):

        # Forward propagation
        z1 = X.dot(W1) + b1
        
        #hyperbolic activation function for layer 1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2

        #hyperbolic activation function for layer 1
        a2 = np.tanh(z2)
        #output is calculuated using linear function
        z3 = a2.dot(W3) + b3
        a3 = z3


        # Backpropagation
        
        #calculating gradients for dA dw and db from output layer
        dA3 = a3 - self.np_1d_to_2d(y)
        dW3 = (a2.T).dot(dA3) / num_examples
        db3 = np.sum(dA3, axis=0, keepdims=True) / num_examples
        
        #calculating gradients for dA dw and db from hidden layer2
        dA2 = dA3.dot(W3.T) * (1 - np.power(a2, 2))
        dW2 = np.dot(a1.T, dA2) / num_examples
        db2 = np.sum(dA2, axis=0, keepdims=True) / num_examples

        #calculating gradients for dA dw and db from hidden layer1

        dA1 = dA2.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, dA1) / num_examples
        db1 = np.sum(dA1, axis=0) / num_examples


        if iteration == 1000 and FLAG_SHOULD_PRINT_ACTIVATIONS:
            print("*** START Demonstrating Regression Neural Network Activation ***\n")
            print("z1 = X@W1 + b1, at hidden layer 1", z1[0])
            print(" Activation, a1 = tanh(z1), at hidden layer 1 for linear regression\n","a1 shape", a1.shape, a1[0])
            print("z2 = a1@W2 + b2, at hidden layer 1", z2[0])
            print(" Activation, a2 = tanh(z2), at hidden layer 2 linear regression\n" ,"a2 shape", a2.shape, a2[0])
            print("final output a3 = a2@W3 + b2, with linear function\n", "a3 shape", a3.shape, a3[0])
            print("\n*** END Demonstrating Regression Neural Network Activation ***\n")

        
        if iteration == 1000 and FLAG_SHOULD_PRINT_WEIGHT_UPDATE:
            print("*** START Demonstrating Linear Neural Net Weight Updates ***\n")
            print("Starting weight W1", W1[0])
            print("weight W1 gradient", dW1[0])
            print("weight updated with learning rate", (W1 - learning_rate * dW1)[0])
            print("Starting weight W2", W2[0])
            print("weight W2 gradient", dW2[0])
            print("weight updated with learning rate", (W2 - learning_rate * dW2)[0])
            print("Starting weight W1", W3[0])
            print("weight W1 gradient", dW3[0])
            print("weight updated with learning rate", (W3 - learning_rate * dW3)[0])
            print("\n*** END Demonstrating Linear Neural Net Weight Updates ***\n")

        if iteration == 1000 and FLAG_SHOULD_PRINT_GRADIENTS:
            print("*** START Demonstrating Gradient Calculation at the Output layer in NN Classification***\n")
            print("dA3", "dA3[range(num_examples), y] -= 1", dA3[0])
            print("dW3", "(a2.T).dot(dA3)", dW3[0])
            print("db3", "np.sum(dA3, axis=0)", db3[0])
            print("\n*** END Demonstrating Gradient Calculation at the Output layer in NN Classification***\n")
          
        # regularization of weights
        dW3 += regularization * W3
        dW2 += regularization * W2
        dW1 += regularization * W1

        # Update weights & biases from gradient
        W1 += -learning_rate * dW1
        b1 += -learning_rate * db1
        W2 += -learning_rate * dW2
        b2 += -learning_rate * db2
        W3 += -learning_rate * dW3
        b3 += -learning_rate * db3
        
        # Assign new parameters to the model
        self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
        
        # compute loss
        if print_loss and iteration % 1000 == 0:
          loss = self.loss(X, y)
          losses.append(loss)
          #print("Loss after iteration %i: %f" %(iteration, loss))
    
    return (self.model, losses)

  #converts np array from 1d to 2d
  def np_1d_to_2d(self, arr):
    return arr.reshape(arr.shape[0],-1)
  
  #compute loss using mean squared error based on the output layer
  def loss(self, X, y):
    """
    Computes loss

    :param X: input data
    :param y: labels
    :return: mean squared error
    """
    W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1

    #hyperbolic activation function for layer 1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    
    #hyperbolic activation function for layer 2
    a2 = np.tanh(z2)

    #final out is calculated using linear function
    z3 = a2.dot(W3) + b3
    a3 = z3

    #loss is calculated using MSE
    loss = np.mean((a3 - self.np_1d_to_2d(y))**2)
    return loss

  def predict(self, X):
    """
    Generates predictions from the model

    :param X: input data
    """
    W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
    # Forward propagation
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)
    z3 = a2.dot(W3) + b3
    a3 = z3
    return a3

  def score(self, X, y):
    """
    Scores the model using input data

    :param X: input data
    :param y: labels
    """
    result = self.predict(X)
    score = self.mse(result, y)
    return score
    
  def mse(self, y_pred, y_actual):
    return np.mean((y_pred - self.np_1d_to_2d(y_actual))**2)


class NeuralNetAutoEncoder(NeuralNet):
  def __init__(self, nn_input_dim, nn_hdim1):
    """
    Initialize weights & biases to random values

    :param nn_input_dim: number of input features
    :param nn_hdim1: dimensions of hidden layer 1
    :return: initialized network
    """
    # Initialize the parameters to random values. We need to learn these.
    #weights are initialized with really small numbers by diving by the sq root of the input dimentions
    #bias is initilized with zeros with dimentions of the respective hidden layer
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim1) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim1))
    W2 = np.random.randn(nn_hdim1, nn_input_dim) / np.sqrt(nn_hdim1)
    b2 = np.zeros((1, nn_input_dim))

    self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

  def fit(self, X, num_iterations, learning_rate=0.1, regularization=0.01, print_loss=False):
    """
    Fits the model from labeled data

    :param X: input data
    :param y: labels
    :param num_iterations: number of iterations
    :param learning_rate: learning rate
    :param regularization: regularization
    :param print_loss: whether to compute and print loss
    :return: model and losses of iterations
    """
    num_examples = len(X)
    
    W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']

    losses = []
    # Gradient descent.
    for iteration in range(0, num_iterations):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
      
        z2 = a1.dot(W2) + b2
        a2 = z2 # linear activation

        if iteration == 1000 and FLAG_SHOULD_PRINT_ACTIVATIONS:
            print("*** START Demonstrating Autoencoder Network Activation ***\n")
            # #print("z1 computation for autoencoder at hidden layer 1 for the first row: z1 = ",reshape_arr(X) * W1 + b1[0])
            # #print("z1 dimentions: " , (reshape_arr(X) * W1 + b1[0]).shape)
            # print("a1 computation for autoencoder at hidden layer 1 for the first row: a1 =  ", np.tanh(reshape_arr(X) * W1+ b1[0]))
            # print("a1 dimentions: ",  np.tanh(reshape_arr(X) * W1 + b1[0]).shape)
            # print("a2 computation for autoencoder at the output layer with linear function: a2 =  ", reshape_arr(a1) * W2 + b2[0])
            # print("a2 dimentions: " , (reshape_arr(a1) * W2 + b2[0]).shape)
            print("z1 = X@W1 + b1, at hidden layer 1", z1[0])
            print("Activation, a1 = tanh(z1) at hidden layer 1 for Autoencoder\n", "a1 shape", a1.shape, a1[0])
            print("Activation, a2 = a1@W2 + b2, at output layer for Autoencoder\n", "a2 shape", a2.shape, a2[0])
            print("\n*** END Demonstrating Autoencoder Network Activation ***\n")

        

        # Backpropagation
        dA3 = a2 - X
        dW2 = (a1.T).dot(dA3) / num_examples
        db2 = np.sum(dA3, axis=0, keepdims=True) / num_examples

        dA2 = dA3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, dA2) / num_examples
        db1 = np.sum(dA2, axis=0) / num_examples


        if iteration == 1000 and FLAG_SHOULD_PRINT_WEIGHT_UPDATE:
            print("*** START Demonstrating Autoencoders Weight Updates ***\n")
            print("Starting weight W1", W1[0])
            print("weight W1 gradient", dW1[0])
            print("weight updated with learning rate", (W1 - learning_rate * dW1)[0])
            print("Starting weight W2", W2[0])
            print("weight W2 gradient", dW2[0])
            print("weight updated with learning rate", (W2 - learning_rate * dW2)[0])
            print("\n*** END Demonstrating Autoencoders Weight Updates ***\n")
          

        # regularization
        dW2 += regularization * W2
        dW1 += regularization * W1

        # Update weights & biases from gradient
        W1 += -learning_rate * dW1
        b1 += -learning_rate * db1
        W2 += -learning_rate * dW2
        b2 += -learning_rate * db2
        
        # Assign new parameters to the model
        self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        # compute loss
        
       
        if print_loss and iteration % 500 == 0:
          loss = self.loss(X)
          losses.append(loss)
          print("Loss after iteration %iteration: %f" %(iteration, loss))
    
    return (self.model, losses)

  def loss(self, X):
    """
    Computes loss

    :param X: input data
    :return: mse
    """
    W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = z2 # linear activation

    loss = self.mse(a2, X)
    return loss

  def predict(self, X):
    """
    Generates predictions from the model

    :param X: input data
    """
    W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
    # Forward propagation
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = z2 # linear activation
    return a2

  def score(self, X):
    """
    Scores the model using input data

    :param X: input data
    """
    result = self.predict(X)
    score = self.mse(result, X)
    return score
    
  def mse(self, X_pred, X_actual):
    return np.mean((X_pred - X_actual)**2)


def neural_network_regression_with_autoencoder(X_train, y_train, X_test, y_test, num_iter_reg, num_iter_encoder, nn_hdim1=20, nn_hdim2=30, **kwargs):
  print("training neural network regression")
  network = NeuralNetRegression(
      nn_input_dim=X_train.shape[1],
      nn_hdim1=nn_hdim1,
      nn_hdim2=nn_hdim2,
      nn_output_dim=1
  )
  #training nn with linear regression with 2 hidden layers
  network.fit(X_train, y_train, num_iter_reg, print_loss=False, **kwargs)
  print("regression train MSE ", network.score(X_train, y_train))
  score = network.score(X_test, y_test)
  print("regression test MSE", score)

  print("training autoencoder")
  #training autoencoder
  autoencoder = NeuralNetAutoEncoder(
      nn_input_dim=X_train.shape[1],
      nn_hdim1=X_train.shape[1] -5, # two fewer features that input dataset to create information bottleneck
  )
  #training autoencoder
  autoencoder.fit(X_train, num_iter_encoder, print_loss=False, **kwargs)
  print("autoencoder train MSE", autoencoder.score(X_train))
  autoencoder_score = autoencoder.score(X_test)
  print("autoencoder test MSE", autoencoder_score)

  print("training neural network regression with autoencode front layer")
  autoencoder_network = NeuralNetRegression(
      nn_input_dim=X_train.shape[1],
      nn_hdim1=X_train.shape[1] -5,
      nn_hdim2=nn_hdim2,
      nn_output_dim=1
  )
  # Put autoencoder's tranform into it's hidden layer as first layer of the normal network
  autoencoder_network.model["W1"] = autoencoder.model["W1"]
  autoencoder_network.model["b1"] = autoencoder.model["b1"]

  #training neural network with one autoencoder and one hidden layer
  autoencoder_network.fit(X_train, y_train, num_iter_reg, print_loss=False, **kwargs)
  print("autoencoder/regression train MSE", autoencoder_network.score(X_train, y_train))
  autoencoder_regression_score = autoencoder_network.score(X_test, y_test)
  print("autoencoder/regression test MSE", autoencoder_regression_score)

  return (score, autoencoder_score, autoencoder_regression_score)

def neural_network_classification_with_autoencoder(X_train, y_train, X_test, y_test, nn_hdim1=20, nn_hdim2=30, **kwargs):
  print("training neural network classification")
  network = NeuralNetClassification(
      nn_input_dim=X_train.shape[1],
      nn_hdim1=nn_hdim1,
      nn_hdim2=nn_hdim2,
      nn_output_dim=len(np.unique(y_train))
  )

  #training nn with classification with 2 hidden layers

  network.fit(X_train, y_train, print_loss=False, **kwargs)
  print("classification train accuracy", network.score(X_train, y_train))
  score = network.score(X_test, y_test)
  print("classification test accuracy", score)

    #training autoencoder
  print("training autoencoder")
  autoencoder = NeuralNetAutoEncoder(
      nn_input_dim=X_train.shape[1],
      nn_hdim1=X_train.shape[1] -2, # two fewer features that input dataset to create information bottleneck
  )
  autoencoder.fit(X_train, print_loss=False, **kwargs)
  #print("autoencoder train MSE", autoencoder.score(X_train))
  autoencoder_score = autoencoder.score(X_test)
  #print("autoencoder test MSE", autoencoder_score)

  print("training neural network regression with autoencode front layer")
  autoencoder_network = NeuralNetClassification(
      nn_input_dim=X_train.shape[1],
      nn_hdim1=X_train.shape[1] -2,
      nn_hdim2=nn_hdim2,
      nn_output_dim=len(np.unique(y_train))
  )
  # Put autoencoder's tranform into it's hidden layer as first layer of the normal network
  autoencoder_network.model["W1"] = autoencoder.model["W1"]
  autoencoder_network.model["b1"] = autoencoder.model["b1"]
  
  #training neural network with one autoencoder and one hidden layer

  autoencoder_network.fit(X_train, y_train, print_loss=False, **kwargs)
  print("autoencoder/classification train accuracy", autoencoder_network.score(X_train, y_train))
  autoencoder_classification_score = autoencoder_network.score(X_test, y_test)
  print("autoencoder/classification test accuracy", autoencoder_classification_score)

  return (score, autoencoder_score, autoencoder_classification_score)

  FLAG_SHOULD_PRINT_ACTIVATIONS = False
FLAG_SHOULD_PRINT_WEIGHT_UPDATE = False
FLAG_SHOULD_PRINT_GRADIENTS = False
def nn_abalone_scores():
  scores = []
  dataset = load_abalone_data()
  folds = split_into_k_folds(dataset, 5, stratify_column="Rings", extract_validation_set=False)
  hidden_layer_1_nodes = [10, 30, 100]
  hidden_layer_2_nodes = [10, 30, 100]
  fold_num = 0
  for fold in folds:
    
    fold_num =fold_num+1
    print("training on fold", fold_num)
    X_train = fold.train.drop("Rings", axis=1).to_numpy()
    y_train= fold.train["Rings"].to_numpy()
    X_test = fold.test.drop("Rings", axis =1).to_numpy()
    y_test= fold.test["Rings"].to_numpy()
    for nodes_1 in hidden_layer_1_nodes:
      for nodes_2 in hidden_layer_2_nodes:
        print("nodes_1 ", nodes_1 ,"nodes_2", nodes_2 )
        score, autoencoder_score, autoencoder_regression_score = neural_network_regression_with_autoencoder(X_train, y_train, X_test, y_test,num_iter_reg = 20000, num_iter_encoder= 12500, nn_hdim1=nodes_1, nn_hdim2=nodes_2)
        lin_scores = {"fold": fold_num, "num_nodes layer 1": nodes_1, "num_nodes layer 2": nodes_2, "score":score, "autoencoder_score":autoencoder_score, "autoencoder_regression_score":autoencoder_regression_score}
        scores.append(lin_scores)
  return scores
abalone_scores = nn_abalone_scores()

def nn_cancer_score():
  scores = []
  dataset = load_breast_cancer_data()
  folds= split_into_k_folds(dataset, 5, stratify_column="Class", extract_validation_set=False)
  hidden_layer_1_nodes = [10, 30, 100]
  hidden_layer_2_nodes = [10, 30, 100]
  fold_num = 0
  for fold in folds:
    fold_num =fold_num+1
    print("training on fold", fold_num)
    X_train = fold.train.drop("Class", axis=1).to_numpy()
    y_train= fold.train["Class"].to_numpy()
    X_test = fold.test.drop("Class", axis =1).to_numpy()
    y_test= fold.test["Class"].to_numpy()
    for nodes_1 in hidden_layer_1_nodes:
      for nodes_2 in hidden_layer_2_nodes:
        print("nodes_1 ", nodes_1 ,"nodes_2", nodes_2 )
        score, autoencoder_score, autoencoder_regression_score = neural_network_classification_with_autoencoder(X_train, y_train, X_test, y_test, nn_hdim1=nodes_1, nn_hdim2=nodes_2)
        lin_scores = {"fold": fold_num, "num_nodes layer 1": nodes_1, "num_nodes layer 2": nodes_2, "score":score, "autoencoder_score":autoencoder_score, "autoencoder_regression_score":autoencoder_regression_score}
        scores.append(lin_scores)
  return scores
cancer_scores = nn_cancer_score()

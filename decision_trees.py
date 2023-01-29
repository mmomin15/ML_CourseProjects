import pandas as pd
import numpy as np
import math
import time
from collections import namedtuple

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
    result[column].fillna(value=result[column].mode(), inplace=True) #fillna applies the mean of a column to cells with NA values
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
   
    return car_data

def load_machine_data():
    machine_data_columns = ["Vendor Name", "Model Name","MYCT","MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    machine_data = load_data ("/content/drive/MyDrive/Colab Notebooks/machine.data",machine_data_columns )
    machine_data

    #one-hot encoding of vendor name
    #machine_data = nominal_categorical_to_one_hot(machine_data, "Vendor Name")
    machine_data = machine_data.drop("Vendor Name", axis = 1)
    #dopping model name as it is defined as non-predictive in the NAME file
    machine_data = machine_data.drop("Model Name", axis = 1)
    
    #standardize numerical columns
    machine_data = standardize_columns(machine_data, ["MYCT","MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"])
    
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
    forest_fires_data = standardize_columns(forest_fires_data, ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "Wind","Rain", "Area"])
    return forest_fires_data

def load_breast_cancer_data():
  # loading breast cancer data and adding column names
    breast_cancer_wisc_data_column_names = ["Sample code number","Clump Thickness","Uniformity of Cell Size", "Uniformity of Cell Shape",
                                       "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli",
                                       "Mitoses", "Class"]
    breast_cancer_wisc_data = load_data("/content/drive/MyDrive/Colab Notebooks/breast-cancer-wisconsin.data", breast_cancer_wisc_data_column_names)
    breast_cancer_wisc_data = fill_missing_with_mode_in_column(breast_cancer_wisc_data, column="Bare Nuclei", downcast='integer')
    #breast_cancer_wisc_data= breast_cancer_wisc_data.astype("int64", errors = "ignore")
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

def train_classification_majority_predictor(train_data, label_column):
    most_common = train_data[label_column].value_counts().nlargest(n=1).index.tolist()[0]
    print(most_common)
    # just representing the model as function instead of a matrix because it's so simple
    def model(test_data):
        print(test_data)
        return pd.Series([most_common for i in range(0, test_data.shape[0])])
    return model
        
def train_regression_majority_predictor(train_data, label_column):
    mean_value = train_data[label_column].mean()
    # just representing the model as function instead of a matrix because it's so simple
    def model(test_data):
        return pd.Series([mean_value for _ in range(0, test_data.shape[0])])
    return model


#get the mode of a column
def most_common(column):
    most_common = column.value_counts().nlargest(n=1).index.tolist()
    return most_common[0]
#get mean
def get_mean(y_data):
    mean_value = y_data.mean()
    return mean_value
#get mean squared error for predicted and actual labels
def mse(y_predicted, y_actual):
    MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
    return MSE
#get root mean squared error
def rmse (y_predicted, y_actual): 
  #uses mean squared error function above
    MSE = mse(y_predicted, y_actual)
    RMSE = math.sqrt(MSE)
    return RMSE


def entropy(col):
   
    unique_vals,counts = np.unique(col,return_counts = True)
    entropy = np.sum([(-counts[instance]/np.sum(counts))*np.log2(counts[instance]/np.sum(counts)) for instance in range(len(unique_vals))])
 
    return entropy

def gain_ratio(data,split_column,target_name="class"):
    #gain ratio function takes in dataset, name of the split column, and target name as arguments
      
    #Calculate the entropy of the original dataset
    initial_entropy = entropy(data[target_name])

    #find the values and counts for the unique values in the split column
    vals,counts= np.unique(data[split_column],return_counts=True)
  
    #Calculate the weighted entropy of the split
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_column]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    #Calculate the information gain by subtracting entropy of the new split
    information_gain = initial_entropy - Weighted_Entropy
   
    #use entropy to calculate split_info
    gain_ratio = 0
    for count in counts:
      split_info = -count/sum(counts)*np.log2(count/np.sum(counts))    
      if split_info > 0:
        #print("split_info", split_info)
      #use information_gain and split_info to calculate gain ratio
        gain_ratio = information_gain/split_info
        #print("gain_ratio", gain_ratio)
      return gain_ratio


def ID3(data, original_data, features, feature_type, target_feature, min_samples, max_depth, parent_node = None, pre_pruning = False ):
    
    #if the number of unique values is < = 1, return the mode of the feature
    if len(np.unique(data[target_feature])) <= 1:
        return np.unique(data[target_feature])[0]

    # if the dataset is empty, use the original dataset to return the mode of the target feature
    elif len(data)==0:
        return np.unique(original_data[target_feature])[np.argmax(np.unique(original_data[target_feature],return_counts=True)[1])]

    #if no features left, return the parent node, that is the mode of the dataset.
    
    elif len(features) ==0:
        return parent_node

    else:
      #parent node  with the mode of target features
      parent_node = np.unique(data[target_feature])[np.argmax(np.unique(data[target_feature],return_counts=True)[1])]
      
      
      #print("parent_node2", parent_node)
      split_gain_ratios = []
      for feature in features:
        #if feature is categorical, split at unique values
        if feature_type[feature] == "cat":
          #calculate gain ratio for categorical values
          split_gain_ratio = gain_ratio(data, feature, target_feature)
          split_gain_ratios.append(split_gain_ratio)
        #if feautre is continuous, discretize the column into bins
        elif feature_type[feature] == "cont":
          data[feature] = pd.to_numeric(data[feature], errors='coerce')
          data[feature] = pd.cut(data[feature], 5, duplicates = 'drop')
          #calculate gain ratio for numerica features after discretization
          split_gain_ratio = gain_ratio(data, feature, target_feature)
          split_gain_ratios.append(split_gain_ratio)
        #get the index of the best feature
      best_feature_idx = np.argmax(split_gain_ratios)
      #get best feature
      best_feature = features[best_feature_idx]

      tree = {best_feature:{}}

      #drop the best feature from the list of feature for further spliting
      features = [i for i in features if i != best_feature]
      
      node = 0
      
      for value in data[best_feature].unique():
            node = node + 1
            value = value
            #Split the dataset along the value of the feature with the largest information gain and therwith create split_datasets
            subset = data.where(data[best_feature] == value).dropna()
            #if pre-pruning is True, use min_samples and max_depth to prune the tree early
            if pre_pruning and (len(subset) < min_samples or node > max_depth): 
                continue
            elif len(subset) >0:
              #Call the recursion here using ID3 function on each of the subsets with new parameters
              tree_node = ID3(subset,original_data,features, feature_type, target_feature, min_samples, max_depth, parent_node, pre_pruning = False)
              #Add the sub tree, grown from the subsetset to the tree under the root node
              tree[best_feature][value] = tree_node
      return (tree)    

def predict_labels(test_data,tree,default = 1):
  for key in list(test_data.keys()):
        #iterates the test_data through the tree and 
        if key in list(tree.keys()):
            #2.
            try:
                result = tree[key][test_data[key]] 
            except:
                return default

            #matches the key of the test data with the key of the tree node
            result = tree[key][test_data[key]]
            #if instance 
            if isinstance(result,dict):
                return predict_labels(test_data,result)

            else:
                return result

def test_predictions(data,tree,target_feature):
  #create new data from the testset by removing the class column, and convert it to dictionary
  test_data = data.drop(target_feature, axis =1).to_dict(orient = "records")
  #test_data = data.iloc[:,:-1].to_dict(orient = "records")
  #Create a empty DataFrame in whose columns the prediction of the tree are stored
  predicted = pd.DataFrame(columns=["predicted"]) 
  #Calculate the prediction accuracy
  for instance in range(len(data)):
      predicted.loc[instance,"predicted"] = predict_labels(test_data[instance],tree,1.0) 
  accuracy = np.sum(predicted["predicted"] == data[target_feature])/len(data)*100
  return accuracy

def post_prune(tree, X_val, target_feature):
  import copy

  # this recursies through a node and gets all leaf node class values so we can compute the mode
  def get_leaf_values_recursive(node):
    result = []
    for feature_name, branches in node.items():
      for class_value, child_node in branches.items():
        if isinstance(child_node, dict): # not a leaf, add all children recursive
          result = result + get_leaf_values_recursive(child_node)
        else: # leaf, just add the one class value
          result = result + [child_node]
    return result
#Reduced Error Pruning
  # recursive function to traverse the tree
  def post_prune_recursive(node):
    for feature_name, branches in node.items():
      for feature_value, child_node in list(branches.items()):
        if isinstance(child_node, dict): # not a leaf
          # do recursion first, this will give us a depth first traversal
          # so we do the bottoms of the free first
          post_prune_recursive(child_node)

          # get all child leaf class values
          child_leaf_values = get_leaf_values_recursive(child_node)
          # if no children don't try to prune
          if child_leaf_values:
            # compte mode, take first if multi-mode
            mode = pd.Series(child_leaf_values).mode()[0]

            # replace the child node with just the mode
            branches['feature_value'] = mode

            # see what the new accuracy
            pruned_accuracy = test_predictions(X_val, tree, target_feature)

            # if the accuracy got worse, put the node back (otherwise leave it removed!)
            if pruned_accuracy < initial_accuracy:
              branches['feature_value'] = child_node

  initial_accuracy = test_predictions(X_val, tree, target_feature)

  tree = copy.deepcopy(tree)
  post_prune_recursive(tree)

  return tree
breast_cancer_data = load_breast_cancer_data()
#drop Sample Code number as it adds to unnecessary complexity as all values are unique, and sample numbers don't determine the outcome.

breast_cancer_data = breast_cancer_data.drop("Sample code number", axis = 1)
cancer_folds= split_into_k_folds(breast_cancer_data, 5, stratify_column="Class", extract_validation_set=True)




FEATURE_DATATYPE_DICT_CANCER = {"Clump Thickness": "cat", "Uniformity of Cell Size": "cat",  "Uniformity of Cell Shape": "cat", 
                          "Marginal Adhesion": "cat", "Single Epithelial Cell Size": "cat", "Bare Nuclei": "cat", "Bland Chromatin": "cat", 
                          "Normal Nucleoli": "cat", "Normal Nucleoli": "cat", "Mitoses": "cat", "Class": "cat"}

#list of tuning parameters for pre-pruning


MIN_SAMPLES = [2 , 5, 10, 100]
MAX_DEPTH = [1, 5, 10, 100]

#tree3 = ID3(X_train_cancer, X_train_cancer, cancer_features, FEATURE_DATATYPE_DICT_CANCER, target_feature="Class", min_samples=10, max_depth = 10, parent_node = None, pre_pruning = True)
fold_num = 0
results_cancer = []
for fold in cancer_folds:
  fold_num= fold_num+1
#get X_train
  X_train_cancer = fold.train
#get Y_train
  y_train_cancer = fold.train["Class"]
#get_x_test
  X_test_cancer = fold.test
#get y_test
  y_test_cancer = fold.test["Class"]
#get X_validatation
  X_validation_cancer = fold.validation
#get y_validation
  y_validation_cancer = fold.validation["Class"]
#get features other than the class column
  cancer_features = X_train_cancer.columns[:-1]
# loop through pre-pruning parameters to test the impact of them on the accuracy of predictions
  for min_samples in MIN_SAMPLES:
    for max_depth in MAX_DEPTH:
      #get the unpruned tree
      cancer_tree = ID3(X_train_cancer, X_train_cancer, cancer_features, FEATURE_DATATYPE_DICT_CANCER, target_feature="Class", min_samples=min_samples, max_depth = max_depth, parent_node = None, pre_pruning = False)
      
      #get the pre-unpruned tree

      cancer_tree_pruned = ID3(X_train_cancer, X_train_cancer, cancer_features, FEATURE_DATATYPE_DICT_CANCER, target_feature="Class", min_samples=min_samples, max_depth = max_depth, parent_node = None, pre_pruning = True)
      
      #get the post-unpruned tree

      cancer_tree_post_pruned = post_prune(cancer_tree, X_validation_cancer, "Class")
     
     #get prediction accuracy of unpruned tree
      cancer_prediction_accuracy = test_predictions(X_test_cancer, cancer_tree, "Class")
      
       #get prediction accuracy of prepruned tree
      cancer_prediction_accuracy_pruned = test_predictions(X_test_cancer, cancer_tree_pruned, "Class")
       #get prediction accuracy of postpruned tree
      cancer_prediction_accuracy_post_pruned = test_predictions(X_test_cancer, cancer_tree_post_pruned, "Class")
      #put it all in results
      result = {"fold": fold_num, "min_samples": min_samples, "max_depth": max_depth, "accuracy no pruning": cancer_prediction_accuracy, "accuracy pre-pruned": cancer_prediction_accuracy_pruned,"accuracy post-pruned": cancer_prediction_accuracy_post_pruned }
      
      print(result)
      #append the results to a list
      
      results_cancer.append(result)
results_cancer

house_votes_data= load_house_votes_data()
house_folds = split_into_k_folds(house_votes_data, 5, stratify_column="Class Name", extract_validation_set=True)

MIN_SAMPLES = [2 , 5, 10, 100]
MAX_DEPTH = [1, 5, 10, 100]

fold_num = 0
house_votes_results = []
for fold in house_folds: 
    fold_num= fold_num+1 
    print(fold_num)
    X_train_house_votes = fold.train
    y_train_house_votes = fold.train["Class Name"]
    X_test_house_votes = fold.test
    y_test_house_votes = fold.test["Class Name"]
    X_validation_house_votes = fold.validation
    y_validation_house_votes = fold.validation["Class Name"]

    FEATURE_DATATYPE_DICT_HOUSE_VOTES ={"Class Name": "cat", "Handicapped-infants": "cat", "water-project-cost-sharing": "cat",
                                    "adoption-of-the-budget-resolution": "cat", "physician-fee-freeze": "cat", "el-salvador-aid": "cat",
                                    "religious-groups-in-schools": "cat", "anti-satellite-test-ban": "cat","aid-to-nicaraguan-contras": "cat", 
                                    "mx-missile": "cat", "immigration": "cat","synfuels-corporation-cutback": "cat","education-spending": "cat",
                                    "superfund-right-to-sue": "cat", "crime": "cat", "duty-free-exports": "cat","export-administration-act-south-africa": "cat"}
    house_votes_features = X_train_house_votes.drop("Class Name", axis =1).columns
    
    for min_sample in MIN_SAMPLES:
      #loop through tuning parameter values
      for max_depth in MAX_DEPTH:

          #get unpruned tree
        house_votes_tree = ID3(X_train_house_votes, X_train_house_votes, house_votes_features, FEATURE_DATATYPE_DICT_HOUSE_VOTES, target_feature="Class Name", min_samples=min_sample, max_depth = max_depth, parent_node = None, pre_pruning = False)
        #get prepruned tree
        house_votes_prepruned_tree = ID3(X_train_house_votes, X_train_house_votes, house_votes_features, FEATURE_DATATYPE_DICT_HOUSE_VOTES, target_feature="Class Name", min_samples=min_sample, max_depth = max_depth, parent_node = None, pre_pruning = True)
        
        house_votes_post_pruned = post_prune(house_votes_tree, X_validation_house_votes, "Class Name")
          #get prediction accuracy of unpruned tree
        house_votes_accuracy = test_predictions(X_test_house_votes, house_votes_tree, "Class Name")
        #get prediction accuracy of prepruned tree
        house_votes_pruned_accuracy = test_predictions(X_test_house_votes, house_votes_prepruned_tree,"Class Name")

        house_votes_prediction_accuracy_post_pruned = test_predictions(X_test_house_votes, house_votes_post_pruned, "Class Name")
        
        result = {"fold": fold_num, "min_samples": min_sample, "max_depth": max_depth, "accuracy no pruning": house_votes_accuracy, "accuracy prepruned": house_votes_pruned_accuracy, "accuracy post pruned ": house_votes_prediction_accuracy_post_pruned }
        print(result)
        house_votes_results.append(result)
house_votes_results

car_data = load_car_data()
car_folds = split_into_k_folds(car_data, 5, stratify_column="Safety", extract_validation_set= True)
#get features other than the class column
FEATURE_DATATYPE_DICT_CAR = {"Buying": "cat", "Maint":"cat", "Doors": "cont", "Persons": "cont", "Lug_boot":"cat", "Safety":"cat"}

MIN_SAMPLES = [2 , 5, 10, 100]
MAX_DEPTH = [1, 5, 10, 100]

fold_num = 0
results_car = []
#create folds
for fold in car_folds:
  fold_num = fold_num + 1
  X_train_car = fold.test
  y_train_car = fold.test["Safety"]
  X_test_car = fold.train
  y_test_car = fold.train["Safety"]
  X_validation_car = fold.validation
  y_validation_car = fold.validation["Safety"]

  car_features = X_train_car.columns[:-1]
  #loop through tuning parameter values
  for min_samples in MIN_SAMPLES:
    for max_depth in MAX_DEPTH:
      car_tree = ID3(X_train_car, X_train_car, car_features, FEATURE_DATATYPE_DICT_CAR, target_feature="Safety", min_samples=min_samples, max_depth = max_depth, parent_node = None, pre_pruning = False)
     #get prediction accuracy of unpruned tree
      car_tree_pruned = ID3(X_train_car, X_train_car, car_features, FEATURE_DATATYPE_DICT_CAR, target_feature="Safety", min_samples=min_samples, max_depth = max_depth, parent_node = None, pre_pruning = True)
      
      car_tree_post_pruned = post_prune(car_tree, X_validation_car, "Safety")
     
      #get prediction accuracy of prepruned tree
      car_prediction_accuracy = test_predictions(X_test_car, car_tree, "Safety")
      car_prediction_accuracy_pruned = test_predictions(X_test_car, car_tree_pruned, "Safety")
      
      car_prediction_accuracy_post_pruned = test_predictions(X_test_car, car_tree_post_pruned, "Safety")

      result = {"fold": fold_num, "min_samples": min_samples, "max_depth": max_depth, "accuracy no pruning": car_prediction_accuracy, "accuracy pre-pruned": car_prediction_accuracy_pruned , "accuracy post-pruned":car_prediction_accuracy_post_pruned}
      print(result)
      results_car.append(result)
results_car




def series_mse(col):
  col_mean = col.mean()
  return ((col - col_mean)**2).mean()

# find best split by tying all split points
def best_split(X_data, y_data):
  #initialize parameters, best_split_col, best_split_point, lowest_mse and best_split
  best_split_col = None
  best_split_point = 0.0
  lowest_mse = 0
  best_split = {
    "left": pd.Series(dtype="float64"),
    "right": pd.Series(dtype="float64"),
  }
  #calculate MSE of the original dataset
  base_mse = series_mse(y_data)
  for column_name in X_data.columns:
    split_points = X_data[column_name].unique()
    for split_point in split_points:
      #splits the data based on unique values in the data; in case of real values, all values
      split = {

        "left": X_data.query(f'`{column_name}` <= {split_point}').index.values,
        "right": X_data.query(f'`{column_name}` > {split_point}').index.values
      }

      if split["left"].shape[0] > 0 and split["right"].shape[0] > 0:
        # Compute the change in loss
        N_left = split["left"].shape[0]
        N_right = split["right"].shape[0]
      
        diff_mse = base_mse - 1.0/y_data.shape[0] * (N_left * series_mse(y_data[split["left"]]) + N_right * series_mse(y_data[split["right"]]))      
        # Update if the change in loss is the largest so far
        if diff_mse >= lowest_mse:
          best_split_col = column_name
          best_split_point = split_point
          best_split = split
          #print("best_split", best_split)
          lowest_mse = diff_mse

  return lowest_mse, best_split_col, best_split_point, best_split

def regression_tree(X_data, y_data, depth = 1, 
              max_depth = 100, tolerance = 10**(-3)):
  node = {}
  # Predict with the mean
  node["weight"] = np.mean(y_data)
 
  node["left"] = None
  node["right"] = None

  # If we can split, find the best split by greedy algorithm
  if y_data.shape[0] >= 2:
    diff_mse, split_column, split_point, split = best_split(X_data, y_data)
    #print("split", split)
    #split again if the stopping criteria has not met
    if split["left"].shape[0] > 0 and split["right"].shape[0] > 0 and diff_mse >= tolerance and depth < max_depth:
      #create a node node["diff_mse"] to store values
      node["diff_mse"] = diff_mse
      #store new split
      node["split_column"] = split_column
      #store new split point
      node["split_point"] = split_point
      #recursively split the data into left and right splits
      node["left"] = regression_tree(X_data.loc[split["left"]], y_data[split["left"]], depth = depth + 1, max_depth = max_depth, tolerance = tolerance)
    
      node["right"] = regression_tree(X_data.loc[split["right"]], y_data[split["right"]], depth = depth + 1, max_depth = max_depth, tolerance = tolerance) 
  return node

def predict_node(node, x):
  #return the weight of the node if no branches on the left
  if node["left"] == None:
    return node["weight"]
    #else split the data on the left
  else:
    try:
      left_of_split_point = x[node["split_column"]] <= node["split_point"]
    except:
      print()
      raise
      #else split the data on the right
    if left_of_split_point:
      return predict_node(node["left"], x)
    else:
      return predict_node(node["right"], x)

def predict_tree(tree, X_data):
  n_instances = X_data.shape[0]
  #create empty list
  preds = np.zeros(n_instances)
  #uses predict_one_node function to predict the whole tree
  for instance in range(0, n_instances):
      preds[instance] = predict_node(tree, X_data.iloc[instance])
  return preds

def print_tree(node, depth = 0):
  #prints tree
  if node["left"] == None:
    print(f'{depth * "  "}weight: {node["weight"]}')
  else:
    print(f'{depth * "  "}X_data{node["split_column"]} <= {node["split_point"]}')
    print_tree(node["left"], depth + 1)
    print_tree(node["right"], depth + 1)


def prediction_mse(tree, X_data, y_data):
    mse = 1/X_data.shape[0]*np.sum(np.square(y_data - predict_tree(tree, X_data)))
    return mse
MAX_DEPTH_FIRES = [10, 100, 500]
TOLERANCE_FIRES = [10**(-1), 10**(-3), 10**(-6) ]



forest_fires_data = load_forest_fires_data()
forest_fires_folds = split_into_k_folds(forest_fires_data, 5, stratify_column="Area", extract_validation_set=False)
fold_num = 0
results_forest_fires =[]
for fold in forest_fires_folds:
  fold_num = fold_num+1
  X_train_forest_fires = fold.train.drop("Area", axis=1)
  y_train_forest_fires= fold.train["Area"]
  X_test_forest_fires = fold.test.drop("Area", axis=1)
  y_test_forest_fires= fold.test["Area"]
  print("fold",fold_num )
  for max_depth in MAX_DEPTH_FIRES:
    for tolerance in TOLERANCE_FIRES:
      tree = regression_tree(X_train_forest_fires, y_train_forest_fires, max_depth = max_depth, tolerance = tolerance)
      predicted_train_mse =  prediction_mse(tree, X_train_forest_fires, y_train_forest_fires)
      predicted_test_mse = prediction_mse(tree, X_test_forest_fires, y_test_forest_fires)
      result = {"fold": fold_num, "max_depth": max_depth, "tolerance": tolerance, "train_mse": predicted_train_mse, "test_mse": predicted_test_mse }
      results_forest_fires.append(result)          
      print(result)
results_forest_fires
#print("results",results_forest_fires)

machine_data = load_machine_data()


MAX_DEPTH_FIRES = [1, 10, 300]
TOLERANCE_FIRES = [10, 10**(-1), 10**(-3) ]



#machine_features = X_train_machine.columns[:-1]
#forest_fires_features

machine_folds = split_into_k_folds(machine_data, 5, stratify_column="ERP", extract_validation_set=False)

fold_num = 0
results_machine = []
for fold in machine_folds:
  fold_num = fold_num+1
  X_train_machine = fold.train.drop("ERP", axis=1)
  y_train_machine = fold.train["ERP"]
  X_test_machine  = fold.test.drop("ERP", axis=1)
  y_test_machine = fold.test["ERP"]
  print("fold",fold_num )
  
  for max_depth in MAX_DEPTH_FIRES:
    for tolerance in TOLERANCE_FIRES:
     
      tree = regression_tree(X_train_machine, y_train_machine, max_depth = max_depth, tolerance = tolerance)
      predicted_train_mse =  prediction_mse(tree, X_train_machine, y_train_machine)
      predicted_test_mse = prediction_mse(tree, X_test_machine, y_test_machine)
      result = {"fold": fold_num, "max_depth": max_depth, "tolerance": tolerance, "train_mse": predicted_train_mse, "test_mse": predicted_test_mse }
      results_machine.append(result)          
    
results_machine

abalone_data = load_abalone_data()


MAX_DEPTH_FIRES = [10, 100, 1000]
TOLERANCE_FIRES = [10**(-1), 10**(-3), 10**(-6) ]

abalone_folds = split_into_k_folds(abalone_data, 5, stratify_column="Rings", extract_validation_set=False)

fold_num = 0
abalone_results =[]
for fold in abalone_folds :
  fold_num = fold_num+1
  X_train_abalone = fold.train.drop("Rings", axis=1)
  y_train_abalone = fold.train["Rings"]
  X_test_abalone  = fold.test.drop("Rings", axis=1)
  y_test_abalone = fold.test["Rings"]
  print("fold",fold_num )
  for max_depth in MAX_DEPTH_FIRES:
    for tolerance in TOLERANCE_FIRES:
     
      tree = regression_tree(X_train_abalone, y_train_abalone, max_depth = max_depth, tolerance = tolerance)
      predicted_train_mse =  prediction_mse(tree, X_train_abalone, y_train_abalone)
      predicted_test_mse = prediction_mse(tree, X_test_abalone, y_test_abalone)
      result = {"fold": fold_num, "max_depth": max_depth, "tolerance": tolerance, "train_mse": predicted_train_mse, "test_mse": predicted_test_mse }
      abalone_results.append(result)          
      print(result)
abalone_results
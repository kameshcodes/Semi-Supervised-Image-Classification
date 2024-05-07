from collections import Counter
import numpy as np


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Decision function to select between left and right nodes.

        Note: In this representation 'True' values for a decision take us to
        the left.

        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.

        Args:
            feature (list(int)): vector for feature.

        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.

    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.

    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if class_index == -1:
        classes = out[:, class_index]
        features = out[:, :class_index]
        return features, classes
    elif class_index == 0:
        classes = out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def custom_criterion(class_vector):
    counts = list(Counter(class_vector).values())
    p_j = [count / len(class_vector) for count in counts]
    K = len(p_j)
    criterion = sum(p * max(0, 1 - (p - 1/K)) for p in p_j)
    return criterion


def information_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    previous_information_gain = custom_criterion(previous_classes)
    current_information_gain = 0
    previous_len = len(previous_classes)
    if len(current_classes[0]) == 0 or len(current_classes[1]) == 0:
        return 0

    for ll in current_classes:
        current_length = len(ll)
        current_information_gain += custom_criterion(ll) * float(current_length) / previous_len

    return previous_information_gain - current_information_gain


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.

        Starts with an empty root.

        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
            depth (int): max depth of tree.  Default is 0.

        Returns:
            Root node of decision tree.
        """
        best_info_gain = -1
        best_column_index = -1
        best_column_threshold = -1
        # Edge Case
        if len(classes) == 0:
            return None

        elif len(classes) == 1:
            return DecisionNode(None, None, None, classes[0])

        elif np.all(classes[0] == classes[:]):
            return DecisionNode(None, None, None, classes[0])

        elif depth == self.depth_limit:
            return DecisionNode(None, None, None, get_most_occurring_feature(classes))

        else:
            # Build tree recursively
            for column_i in range(features.shape[1]):
                column_values_for_column_i = features[:, column_i]
                column_mean = np.mean(column_values_for_column_i)

                classes_new = []
                temp_X_left, temp_X_right, temp_y_left, temp_y_right = partition_classes(features, classes, column_i,
                                                                                         column_mean)
                classes_new.append(temp_y_left)
                classes_new.append(temp_y_right)
                column_i_information_gain = information_gain(classes, classes_new)

                # SETUP BEST MATRIX
                if column_i_information_gain > best_info_gain:
                    best_info_gain = column_i_information_gain
                    best_column_index = column_i
                    best_column_threshold = column_mean

            # now we have found the best column and the associated properties, lets now divide the data set
            X_left, X_right, y_left, y_right = partition_classes(features, classes, best_column_index,
                                                                 best_column_threshold)
            depth += 1

            left_tree = self.__build_tree__(np.array(X_left), np.array(y_left), depth)
            right_tree = self.__build_tree__(np.array(X_right), np.array(y_right), depth)

            return DecisionNode(left_tree, right_tree,
                                lambda feature: feature[best_column_index] < best_column_threshold)

    def classify(self, features):

        class_labels = []

        for feature in features:
            tree = self.root
            class_labels.append(tree.decide(feature))
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.

    Randomly split data into k equal subsets.

    Fold is a tuple (training_set, test_set).
    Set is a tuple (examples, classes).

    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.

    Returns:
        List of folds.
    """
    final_answer = []
    fold_size = len(dataset[1]) // k
    num_folds = k
    remaining_indexes_to_select_from = np.arange(len(dataset[1]))
    X = dataset[0]
    y = dataset[1]
    for i in range(num_folds):
        # generate first fold
        indexes_chosen_for_this_fold = []
        for l in range(fold_size):
            random_index = np.random.choice(remaining_indexes_to_select_from)
            temp_list = remaining_indexes_to_select_from.tolist()
            temp_list.remove(random_index)
            remaining_indexes_to_select_from = np.asarray(temp_list)
            indexes_chosen_for_this_fold.append(random_index)

        testing_set = (X[indexes_chosen_for_this_fold], list(y[indexes_chosen_for_this_fold]))
        training_set_X = np.delete(X, indexes_chosen_for_this_fold, 0)
        training_set_Y = list(np.delete(y, indexes_chosen_for_this_fold))
        training_set = (training_set_X, training_set_Y)
        # Set is a tuple (examples, classes).

        final_answer.append((training_set, testing_set))

    return final_answer


class RandomForest:
    def __init__(self, num_trees, depth_limit, example_subsample_rate, attr_subsample_rate):
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.out_of_bag_samples = [[] for _ in range(self.num_trees)]

    def fit(self, features, classes):
        for i in range(self.num_trees):
            actual_example_num = int(features.shape[0] * self.example_subsample_rate)
            actual_attribute_num = int(features.shape[1] * self.attr_subsample_rate)
            chosen_features = []
            chosen_classes = []
            chosen_indices = []  

            for _ in range(actual_example_num):
                random_index = np.random.randint(0, features.shape[0])
                chosen_features.append(features[random_index])
                chosen_classes.append(classes[random_index])
                chosen_indices.append(random_index)  

            chosen_column_values = set()
            total = 0
            while total < actual_attribute_num:
                random_column_chosen = np.random.randint(0, features.shape[1])
                if random_column_chosen not in chosen_column_values:
                    chosen_column_values.add(random_column_chosen)
                    total += 1

            dt = self.__build_tree__((np.asarray(chosen_features)[:, list(chosen_column_values)]), np.asarray(chosen_classes), 0)
            self.trees.append(dt)
            out_of_bag_indices = [idx for idx in range(features.shape[0]) if idx not in chosen_indices]
            self.out_of_bag_samples[i] = out_of_bag_indices

    def __build_tree__(self, features, classes, depth=0):
        best_info_gain = -1
        best_column_index = -1
        best_column_threshold = -1

        if len(classes) == 0:
            return None
        elif len(classes) == 1:
            return DecisionNode(None, None, None, classes[0])
        elif np.all(classes[0] == classes[:]):
            return DecisionNode(None, None, None, classes[0])
        elif depth == self.depth_limit:
            return DecisionNode(None, None, None, get_most_occurring_feature(classes))
        else:
            for column_i in range(features.shape[1]):
                column_values_for_column_i = features[:, column_i]
                column_mean = np.mean(column_values_for_column_i)
                classes_new = []
                temp_X_left, temp_X_right, temp_y_left, temp_y_right = partition_classes(features, classes, column_i, column_mean)
                classes_new.append(temp_y_left)
                classes_new.append(temp_y_right)
                column_i_information_gain = information_gain(classes, classes_new)
                if column_i_information_gain > best_info_gain:
                    best_info_gain = column_i_information_gain
                    best_column_index = column_i
                    best_column_threshold = column_mean
            X_left, X_right, y_left, y_right = partition_classes(features, classes, best_column_index, best_column_threshold)
            depth += 1
            decision_function = lambda feature: feature[best_column_index] < best_column_threshold
            if len(y_left) == 0:
                decision_function = lambda feature: False
            if len(y_right) == 0:
                decision_function = lambda feature: True
            right_tree = self.__build_tree__(np.array(X_right), np.array(y_right), depth)
            left_tree = self.__build_tree__(np.array(X_left), np.array(y_left), depth)
            return DecisionNode(left_tree, right_tree, decision_function)

    def classify(self, features):
        predictions = []
        for feature in features:
            decisions = []
            for tree in self.trees:
                decisions.append(tree.decide(feature))
            predictions.append(get_most_occurring_feature(decisions))
        return predictions

    def oob_score(self, features, classes):
        correct_predictions = 0
        total_predictions = 0

        for i in range(len(features)):
            oob_trees = [tree_idx for tree_idx, sample_idx_list in enumerate(self.out_of_bag_samples) if i not in sample_idx_list]
            if len(oob_trees) == 0:
                continue

            predictions = [self.trees[tree_idx].decide(features[i]) for tree_idx in oob_trees]
            predicted_class = max(set(predictions), key=predictions.count)

            if predicted_class == classes[i]:
                correct_predictions += 1
            total_predictions += 1

        if total_predictions == 0:
            return 0.0
        else:
            return correct_predictions / total_predictions

def partition_classes(X, y, split_attribute, split_val):
    X_left = []
    X_right = []

    y_left = []
    y_right = []

    for i in range(len(X)):
        if float(X[i][split_attribute]) <= split_val:
            X_left.append(X[i])
            y_left.append(y[i])
        else:
            X_right.append(X[i])
            y_right.append(y[i])

    return X_left, X_right, y_left, y_right

def get_most_occurring_feature(classes):
    counter = Counter(classes)
    k, v = counter.most_common(1)[0]
    return k

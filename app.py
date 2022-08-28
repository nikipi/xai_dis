

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd
#
# print(observe_prepare_data('heloc_dataset_v1.csv'))
#
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import numpy as np
import re
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from IPython.display import display
#
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import shap
from sklearn.tree import _tree

# choose a random element from a list
from random import seed
from random import choice
import math
#

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.metrics import classification_report, confusion_matrix

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)



def observe_prepare_data(filename):
    data = pd.read_csv(filename, engine='python', index_col=False)
    # remove first column
    data = data.iloc[:, 1:]
    # print(data.head())
    # print(data.info())
    # print(data.describe())
    # print(data.shape)

    ### drop special cases
    cols_name = data.columns
    for cols in cols_name:
        data = data[data[cols] != -9]
        data = data[data[cols] != -8]
        data = data[data[cols] != -7]

    print('Special cases have been dropped')

    variable_names = list(data.columns[1:])
    print(variable_names)
    X = data[variable_names]

    data['RiskPerformance'] = np.where(data['RiskPerformance'] == "Bad", 1, 0)

    y = data['RiskPerformance']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=83)

    return X, y, X_train, X_test, y_train, y_test



class Model():
    def __init__(self, file_name):
        self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test = observe_prepare_data(file_name)
        print(self.y_train)
        self.model = None

    def train_model(self, max_depth=3):
        self.model = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=50, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def test_model(self):
        if (not self.model):
            print("Train Model First")

        self.train_pred = self.model.predict(self.X_train)
        self.test_pred = self.model.predict(self.X_test)

        self.y_train_df = pd.DataFrame(self.y_train)
        # self.train_pred_df = pd.DataFrame(self.train_pred, columns= ['Predict'])
        self.y_train_df['Predict'] = self.train_pred

        self.y_test_df = pd.DataFrame(self.y_test)
        # self.train_pred_df = pd.DataFrame(self.train_pred, columns= ['Predict'])
        self.y_test_df['Predict'] = self.test_pred

        acc_train = round((np.mean(self.train_pred == self.y_train) * 100), 2)
        acc_test = round((np.mean(self.test_pred == self.y_test) * 100), 2)

        self.train = pd.concat([self.X_train, self.y_train_df], axis=1)
        self.test = pd.concat([self.X_test, self.y_test_df], axis=1)

        self.test.to_csv('heloc_dataset_test.csv')

        print("Training Accuracy:", acc_train, '%')
        print("Test Accuracy:", acc_test, '%')



    def show_tree(self):

        dot_data = tree.export_graphviz(self.model, out_file=None,
                                        feature_names=self.X_train.columns,
                                        class_names=['Good', 'Bad'], filled=True,
                                        rounded=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render("Gini")
        graph.view()

    def findpath(self):
        feature = self.model.tree_.feature
        decision_path = {}
        max_depth = 3

        decisionpath = []
        a = []
        for i in feature:
            if i > -2:
                a.append(i)
            else:
                if a:  # make sure a has something in it
                    decisionpath.append(a)
                a = []
        if a:  # if a is still accumulating elements
            decisionpath.append(a)

        max_depth = 3

        for pathnum in range(0, len(decisionpath)):
            if len(decisionpath[pathnum]) < 3:
                decisionpath[pathnum] = decisionpath[pathnum - 1][:max_depth - len(decisionpath[pathnum])] + \
                                        decisionpath[pathnum]
        col_name = self.X_train.columns

        pathname = []

        for path in decisionpath:
            eachpath = []
            for i in path:
                eachpath.append(self.X_train.columns[i])
            pathname.append(eachpath)

        self.decisionpath = pathname

    def findpaththred(self):
        self.threshold = self.model.tree_.threshold
        decision_thre = {}
        max_depth = 3

        decisionthred = []
        a = []
        for i in self.threshold:
            if i > -2:
                a.append(i)
            else:
                if a:  # make sure a has something in it
                    decisionthred.append(a)
                a = []
        if a:  # if a is still accumulating elements
            decisionthred.append(a)

        max_depth = 3

        for pathnum in range(0, len(decisionthred)):
            if len(decisionthred[pathnum]) < 3:
                decisionthred[pathnum] = decisionthred[pathnum - 1][:max_depth - len(decisionthred[pathnum])] + \
                                         decisionthred[pathnum]

        self.decisionthred = decisionthred

    def show_rules(self, class_names):

        tree_ = self.model.tree_
        feature_name = [
            self.X_train.columns[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature]

        paths = []
        path = []

        def recurse(node, path, paths):

            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                p1 += [f"({name} <= {np.round(threshold, 3)})"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [f"({name} > {np.round(threshold, 3)})"]
                recurse(tree_.children_right[node], p2, paths)
            else:
                path += [(tree_.value[node], tree_.n_node_samples[node])]
                paths += [path]

        recurse(0, path, paths)

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]

        self.rules = []

        for path in paths:
            print(path)
            rule = "if "

            for p in path[:-1]:
                if rule != "if ":
                    rule += " and "
                rule += str(p)
            # rule += " then "
            # if class_names is None:
            #
            #     rule += "response: " + str(path[-1][0][0][0])
            # else:
            #     classes = path[-1][0][0]
            #     l = np.argmax(classes)
            #     rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
            rule += f" | based on {path[-1][1]:,} samples"
            self.rules += [rule]

        return self.rules

    def dataframeforeachnode(self):

        list_of_dfs = []

        for rule in self.rules:

            pat = '\(.*?\)'
            simple_rules = re.findall(pat, rule)
            var = []
            direction = []
            thre = []
            for simple_rule in simple_rules:
                rule = simple_rule.split(' ')
                var.append(rule[0][1:])
                direction.append(rule[1])
                thre.append(rule[2][:-1])

            print(var, direction, thre)
            split_rules = ''
            for i in range(0, len(var)):
                if i != len(var) - 1:
                    split_rule = var[i] + direction[i] + thre[i]
                    split_rule = '(' + split_rule + ')' + '&'

                else:
                    split_rule = var[i] + direction[i] + thre[i]
                    split_rule = '(' + split_rule + ')'
                split_rules += split_rule
            print(split_rules)
            split_df = self.test.query(split_rules)
            list_of_dfs.append(split_df)

        self.list_of_dfs = list_of_dfs

    def leafaccuracy(self):
        self.accuracy = []
        for i in self.list_of_dfs:
            self.accuracy.append(round(np.mean(i['RiskPerformance'].array == i['Predict'].array) * 100, 2))

    def findtestleaf(self):
        leafdis = self.model.apply(self.X_test)

        leaf_order = {3: 0, 4: 0, 6: 2, 7: 3, 10: 4, 11: 5, 13: 6, 14: 7}

        newlis = [leaf_order[item] for item in leafdis]

        self.test_leaf = newlis

    def returnleafnoed(self, idnum):

        st.write('the leaf node is', self.test_copy.iloc[idnum]['leafid'])

    def test_leaf_node(self):
        self.test_copy = self.test.copy()
        self.test_copy['leafid'] = self.test_leaf


    def shapfeatureimportance(self, idnum):
        self.explainer = shap.TreeExplainer(self.model)
        data_for_prediction = self.X_test.iloc[idnum]
        self.features = self.X_test.columns
        # self.glo_shap = self.explainer.shap_values(self.X_test)[0]
        # print(len(self.glo_shap))

        if self.test.iloc[idnum]['Predict'] == 1:
            self.shap_values = self.explainer.shap_values(data_for_prediction)

            indexes = [index for index, value in enumerate(self.shap_values[1]) if value > 0]
            self.makeitfeature = [self.features[index] for index in indexes]
            st.write('The model predicts the person as bad based on', self.makeitfeature)

        else:
            self.shap_values = self.explainer.shap_values(data_for_prediction)
            indexes = [index for index, value in enumerate(self.shap_values[0]) if value > 0]
            self.makeitfeature = [self.features[index] for index in indexes]
            st.write('The model predicts the person as good based on', self.makeitfeature)

            shap.initjs()
            display(shap.force_plot(self.explainer.expected_value[1], self.shap_values[1], data_for_prediction))

    def plotdisforimportantfeature(self, idnum):

        #         color = ['red','blue','green','black','brown']

        #         fig,ax = plt.subplots(len(self.makeitfeature),sharex=True,sharey=True,figsize=(10,10))

        #         plt.suptitle('Model')
        #         plt.xlabel('Feature')
        #         plt.ylabel('Class')

        #         for i in range(len(self.makeitfeature)):
        #             X = self.X.loc[:,self.makeitfeature[i]]
        #             Y = self.y
        #             ax[i].scatter(X, Y,color= color[i],label=self.makeitfeature[i])
        #             plt.xlim([max(X), min(X)])
        #             ax[i].legend(loc=4, prop={'size': 8})
        #             ax[i].set_title('Distribution for feature %s'% self.makeitfeature[i])

        #         fig1 = px.histogram(df, x=feature, color="Diagnosis", marginal="rug")
        #         plotly_chart(fig1,use_container_width=True)

        sns.set(rc={'figure.figsize': (10, 8)})
        for i in self.makeitfeature:
            featurevalue = self.test[i].iloc[idnum]
            print(featurevalue)

            sns.histplot(data=self.train
                         , x=i
                         , alpha=.9
                         , hue='RiskPerformance'
                         )

            plt.vlines(featurevalue, 0, 2, color='r', label='mean', colors="r", linewidth=8)
            plt.annotate('The feature value for this client is %d' % featurevalue, xy=(featurevalue, 0),
                         xytext=(featurevalue + 3, 40),
                         arrowprops=dict(facecolor='black',
                                         connectionstyle="arc3,rad=-0.2"), textcoords='offset points',
                         bbox=dict(boxstyle="round4,pad=.5", fc="0.9"))

            #             arrowprops=dict(facecolor='black', shrink=0.05))

            st.pyplot(plt)
            plt.cla()


    def testidlist(self):
        # seed random number generator
        seed(1)
        # prepare a sequence
        idsequence = [i for i in range(500)]
        self.idlist = []
        # make choices from the sequence
        for _ in range(20):
            self.idlist.append(choice(idsequence))

    def method_one_only_show_prediction(self, idnum):
        #         self.methodonepred = []


        if self.test['Predict'].iloc[idnum] == 0:
            st.write(
                'The model predicts that this client has low risk of not paying the loan, and suggests you accept his/her application')
        else:
            st.write(
                'The model predicts that this client has high risk of not paying the loan, and suggests you reject his/her application')

    def method_two_show_decision_path_and_it_accuracy(self, idnum):
        #         self.methodtwopred = []
        #         for i in self.idlist:
        leafid = self.test_copy.iloc[idnum]['leafid']
        st.write('The information for id number %d is' % idnum, self.X_test.iloc[idnum, :])
        st.write('the leaf node is ', leafid)

        if self.test['Predict'].iloc[idnum] == 0:
            st.write(
                'The model predicts that this client has low risk of not paying the loan using the rule:',self.rules[leafid],'and suggests you accept his/her application')
        else:
            st.write(
                'The model predicts that this client has high risk of not paying the loan using the rule:', self.rules[leafid],'and suggests you reject his/her application')

        st.write('The accuracy of this decision path is:', self.accuracy[leafid])

    #             self.methodtwopred.append(self.test['Predict'].iloc[i])

    def method_three_show_importance_feature(self):
        for i in self.idlist:
            self.shapfeatureimportance(i)

    #             self.plotdisforimportantfeature(i)

    def find_nearest_neighbor(self, df, i, num_neighbors):
        distances = list()
        for j in range(len(df)):
            dist = euclidean_distance(df.iloc[j, :], self.X_test.iloc[i, :])
            distances.append((df.iloc[j], dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        knn = pd.DataFrame(neighbors)
        index = knn.index

        return index

    def example_base(self, idnum):

        #             print('The information for id number is')
        #             print(pd.DataFrame( self.X_test.iloc[[i]]) )
        idn = list(self.find_nearest_neighbor(self.X_train, idnum, 3))
        print(idn)

        st.write('The nearest neighbors are:')
        newdf = self.train[self.train.index.isin(idn)]
        st.dataframe(newdf)



if __name__ == "__main__":

    a= Model('/Users/ypi/Desktop/xai_data/heloc_dataset_use.csv')
    a.train_model(3)
    a.test_model()

    a.show_rules(None)
    a.dataframeforeachnode()
    a.leafaccuracy()
    a.findtestleaf()
    a.test_leaf_node()




    testid=[54]
    #,54,411,192,463,528,384,395,408,417

    for idnum in testid:
        st.write('The information for id number %d is' % idnum, a.X_test.iloc[idnum, :])
        st.write('The prediction is ', a.test.iloc[idnum]['Predict'])
        st.write('The reality is ', a.test.iloc[idnum]['RiskPerformance'])

        method = st.selectbox(
        'Choose your method ', ('Feature-Based-Explanation', 'Rule-Based-Explanation',
                                'Example-Based-Explanation'))

        if method == 'Feature-Based-Explanation':
            a.method_one_only_show_prediction(idnum)
            a.shapfeatureimportance(idnum)
            a.plotdisforimportantfeature(idnum)
        if method == 'Rule-Based-Explanation':
            a.method_two_show_decision_path_and_it_accuracy(idnum)
        if method == 'Example-Based-Explanation':
            a.example_base(idnum)



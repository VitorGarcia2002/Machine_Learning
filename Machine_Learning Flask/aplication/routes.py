from flask import render_template, request
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from joblib import dump
import pandas as pd
import matplotlib
matplotlib.use('Agg')



df = pd.read_csv('C:\\Machine Learning\\Machine_Learning Flask\\aplication\\data\\Breast_Cancer.csv')
df.isnull().sum()
df = df.drop("Unnamed: 32",axis=1)
df['diagnosis'] = df['diagnosis'].replace({'B':0,'M':1})
df['diagnosis'].value_counts()
df = df.drop("id",axis=1)
X = df.drop("diagnosis",axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=7)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from aplication import app

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train_knn', methods=['GET'])
def train_knn_route():
    return render_template('knn.html')

@app.route('/show_results_knn', methods=['POST'])
def show_results_knn():
    n_neighbors = int(request.form.get('n_neighbors', 5))
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(X_train, y_train)

    dump(classifier, 'C:\\Machine Learning\\Machine_Learning Flask\\aplication\\models\\knn_model.pkl')

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    cm = confusion_matrix(y_test, y_pred)
    
    plt.switch_backend('Agg')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('result.html', classifier='KNN', accuracy=accuracy, macro_f1=macro_f1, plot_url=plot_url)


@app.route('/train_decision_tree', methods=['GET'])
def train_decision_tree_route():
    return render_template('decision_tree.html')


@app.route('/show_results_decision_tree', methods=['POST'])
def show_results_decision_tree():
    max_depth = int(request.form.get('max_depth', 3))
    min_samples_split = int(request.form.get('min_samples_split', 2))
    min_samples_leaf = int(request.form.get('min_samples_leaf', 1))

    classifier = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    classifier.fit(X_train, y_train)

    dump(classifier, 'C:\\Machine Learning\\Machine_Learning Flask\\aplication\\models\\decision_tree_model.pkl')

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    cm = confusion_matrix(y_test, y_pred)

    plt.switch_backend('Agg')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('result.html', classifier='Decision Tree', accuracy=accuracy, macro_f1=macro_f1, plot_url=plot_url)

@app.route('/train_svm', methods=['GET'])
def train_svm_route():
    return render_template('svm.html')

# Rota para exibir os resultados do SVM
@app.route('/show_results_svm', methods=['POST'])
def show_results_svm():
    C = float(request.form.get('C', 1.0))
    kernel = request.form.get('kernel', 'rbf')
    gamma = request.form.get('gamma', 'scale')

    classifier = SVC(C=C, kernel=kernel, gamma=gamma)
    classifier.fit(X_train, y_train)

    dump(classifier, 'C:\\Machine Learning\\Machine_Learning Flask\\aplication\\models\\svm_model.pkl')

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    cm = confusion_matrix(y_test, y_pred)

    plt.switch_backend('Agg')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('result.html', classifier='SVM', accuracy=accuracy, macro_f1=macro_f1, plot_url=plot_url)



@app.route('/train_random_forest', methods=['GET'])
def train_random_forest_route():
    return render_template('random_forest.html')

# Rota para exibir os resultados do Random Forest
@app.route('/show_results_random_forest', methods=['POST'])
def show_results_random_forest():
    n_estimators = int(request.form.get('n_estimators', 100))
    criterion = request.form.get('criterion', 'gini')
    max_depth = int(request.form.get('max_depth', None))
    min_samples_split = int(request.form.get('min_samples_split', 2))
    min_samples_leaf = int(request.form.get('min_samples_leaf', 1))

    # Treine o modelo
    classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    classifier.fit(X_train, y_train)

    dump(classifier, 'C:\\Machine Learning\\Machine_Learning Flask\\aplication\\models\\random_forest_model.pkl')

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    cm = confusion_matrix(y_test, y_pred)

    plt.switch_backend('Agg')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('result.html', classifier='Random Forest', accuracy=accuracy, macro_f1=macro_f1, plot_url=plot_url)
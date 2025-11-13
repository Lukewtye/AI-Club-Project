import matplotlib.pyplot as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo
import ydata_profiling as yp 
from sklearn.datasets import load_breast_cancer
import seaborn as sns

"""
pip install -r requirements.txt to install dependencies, use version > 3.13 of python
"""

# set plot style
plt.style.use("ggplot")

def main(): 
    dataset = load_breast_cancer()
    feature_names = dataset.feature_names 
    target_names = dataset.target_names 
    X = dataset.data 
    y = dataset.target 
  
    #Defining Splits for Test/Train & Stratified data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1, shuffle=True, stratify=y)

    #Model - Stochastic Gradient Descent Classifier
    from sklearn.svm import SVC
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report, roc_auc_score

    #Constructing Pipelin with scaler and svc
    pipe = Pipeline([('scaler', StandardScaler()), ('sgd', SGDClassifier())])
    #Setting Parameters for model fitting
    pipe.set_params(
        sgd__loss="log_loss", 
        sgd__penalty="l2", 
        sgd__early_stopping=True, 
        sgd__validation_fraction=0.1, 
        sgd__n_iter_no_change=5)
    
    #Fitting Model and Scoring
    pipe.fit(X_train, y_train)
    print(("Test accuracy: (SGD)"), pipe.score(X_test, y_test), "%")

def basedatainfo():
    dataset = load_breast_cancer()
    x = dataset.data 
    y = dataset.target 
    
    feature_names = dataset.feature_names 
    target_names = dataset.target_names 
    
    print("Feature names:", feature_names) 
    print("Target names:", target_names) 
    print("\nType of X is:", type(x)) 
    print("\nFirst 5 rows of X:\n", x[ :5])

def graphing_datasets():
    #Base Code
    dataset = load_breast_cancer()
    feature_names = dataset.feature_names 
    target_names = dataset.target_names 
    X = dataset.data 
    y = dataset.target 

    # Convert to DataFrame for easier handling
    X = pd.DataFrame(X, columns = feature_names)
    y = pd.Series(y)

    #Histogram Code
    df = pd.concat([X, y], axis= 1)
    df.head()
    sns.histplot(df["mean radius"], kde=True)
    plt.title("Distribution of Radius Mean")
    plt.show()

    #Q-Q Plot Code
    from scipy import stats

    stats.probplot(df["mean radius"], dist="norm", plot=plt)
    plt.title("Q–Q Plot: Radius Mean")
    plt.show()

    #B/M Graph 
    # Map numeric targets to text labels
    mapping = {i: name.capitalize() for i, name in enumerate(dataset.target_names)}
    y_named = y.map(mapping)

    # Count and plot with names instead of 0/1
    target_counts = y_named.value_counts()

    ax = target_counts.plot(
        kind="bar",
        title="Benign vs Malignant Tumors",
        xlabel="", 
        ylabel="Count",
        color=["lightcoral", "lightgreen"]
    )
    ax.set_xticklabels(target_counts.index, rotation=0, ha="center")
    plt.show()

    
if __name__ == "__main__":
    main()
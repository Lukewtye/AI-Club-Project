import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

plt.style.use('ggplot')
def main():
    # fetch dataset 
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
    
    # data (as pandas dataframes) 
    X = breast_cancer_wisconsin_diagnostic.data.features 
    y = breast_cancer_wisconsin_diagnostic.data.targets 
    
    # metadata 
    print(breast_cancer_wisconsin_diagnostic.metadata) 
    
    # variable information 
    print(breast_cancer_wisconsin_diagnostic.variables) 

if __name__ == "__main__":
    main()
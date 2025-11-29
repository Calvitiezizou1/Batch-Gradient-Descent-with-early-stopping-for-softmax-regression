import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

np.random.seed(42)

class SoftmaxRegressor : 
    def __init__(self) : 
        
        self.X = None
        self.y = None 
        self.X_with_bias = None 
        self.X_train = None
        self.y_train = None 
        self.X_test = None 
        self.y_test = None 
        self.X_valid = None 
        self.y_valid = None 
        self.y_one_hot = None
        self.y_test_one_hot = None
        self.y_valid_one_hot = None
        self.Theta = None
        self.best_theta = None
        self.best_losses = None
        self.best_epoch = None
        self.n_input = None
        self.n_output = None
        self.mean  = None
        self.std = None
        self.train_loss_history = []  # Pour stocker la perte d'entraînement
        self.valid_loss_history = []  # Pour stocker la perte de validation


    def load_data(self): 
        """ Le terme de biais  agit comme une constante qui ajuste le score de chaque classe indépendamment des caractéristiques\. 
        Sans ce terme, le modèle suppose que la frontière de décision passe par l'origine ,
        ce qui peut être une contrainte trop forte et réduire la capacité du modèle à bien séparer les classes."""
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        self.X = iris.data[["petal length (cm)", "petal width (cm)"]].values

    
        self.X_with_bias = np.c_[np.ones(len(self.X)), self.X]

        self.y = iris["target"].values
   
    def preprocess(self): 
        ## TRAIN TEST SPLIT
        test_ratio = 0.2
        valid_ratio = 0.2
        total_size = len(self.X)
        test_len = int(test_ratio * total_size)
        valid_len = int(valid_ratio * total_size)
        train_len = total_size - test_len - valid_len

        
        rnd_permutation = np.random.permutation(total_size)

        self.X_train = self.X_with_bias[rnd_permutation[:train_len]]
        self.y_train = self.y[rnd_permutation[:train_len]]
        self.X_test = self.X_with_bias[rnd_permutation[train_len:train_len + test_len]]
        self.y_test = self.y[rnd_permutation[train_len:train_len + test_len]]
        self.X_valid = self.X_with_bias[rnd_permutation[train_len + test_len:]]
        self.y_valid = self.y[rnd_permutation[train_len + test_len:]]

        ## STANDARD SCALER 
        self.mean = np.mean(self.X,axis = 0)
        self.std = np.std(self.X , axis =0 )

        self.X_train[:,1:] = (self.X_train[:,1:] - self.mean) /self.std
        self.X_valid[:,1:] = (self.X_valid[:,1:] - self.mean) /self.std
        self.X_test[:,1:]  = (self.X_test[:,1:] - self.mean) /self.std


    def one_hot(self,y) : 
        return np.diag(np.ones(self.y.max() + 1))[y]

    def softmax(self, logits): 
        exps = np.exp(logits)
        return exps/exps.sum( axis=1 , keepdims= True)
    
    # Algorithme de descente de gradient : theta = theta - eta * Gradient(Fonction de cout)    


    def gradient_descent(self, eta = 0.5 , n_epoches = 5001):
        """Implementation de la descente de gradient"""
        self.n_input = 3
        self.n_output = 3 
        self.Theta  = np.random.randn(self.n_input,self.n_output)
        self.y_train_one_hot = self.one_hot( self.y_train)
        self.y_test_one_hot = self.one_hot(self.y_test)
        self.y_valid_one_hot = self.one_hot(self.y_valid)
        epsilon = 1e-5

        m = len(self.X_train)
        print("\n\n")
        for epoch in range(n_epoches):
            logits = self.X_train @ self.Theta 
            y_proba = self.softmax(logits)
                
            train_entropy = self.y_train_one_hot * np.log(y_proba + epsilon)      
            train_cross_entropy = - np.sum(train_entropy,axis=1).mean()  
            self.train_loss_history.append(train_cross_entropy)

            y_valid_proba = self.softmax(self.X_valid @ self.Theta)
            valid_entropy = self.y_valid_one_hot * np.log(y_valid_proba +epsilon)
            valid_cross_entropy = -np.sum(valid_entropy , axis = 1).mean()
            self.valid_loss_history.append(valid_cross_entropy)
            if epoch %1000 == 0 : 
                print(f" epoque : {epoch}, entropie coisé :{valid_cross_entropy} ")

            error = y_proba - self.y_train_one_hot 
            gradient = 1/ m * self.X_train.T @ error 

            self.Theta = self.Theta - eta * gradient
        print(f"\n\nParrametres du model : {self.Theta}")


    def grad_desc_go_back(self , eta = 0.5 , n_epoches = 5000):
        print("\n")
        self.n_output = 3 
        self.n_input = 3
        self.Theta  = np.random.randn(self.n_input,self.n_output)
        self.y_train_one_hot = self.one_hot( self.y_train)
        self.y_test_one_hot = self.one_hot(self.y_test)
        self.y_valid_one_hot = self.one_hot(self.y_valid)
        self.best_losses =100
        epsilon = 1e-5
        alpha = 1

        m = len(self.X_train)
        
        for epoch in range(n_epoches):
            logits = self.X_train @ self.Theta 
            y_proba = self.softmax(logits)
            
            # cross-entropy : train et valid
            train_entropy = self.y_train_one_hot * np.log(y_proba + epsilon)      
            train_cross_entropy = - np.sum(train_entropy,axis=1).mean()  
            self.train_loss_history.append(train_cross_entropy)

            y_valid_proba = self.softmax(self.X_valid @ self.Theta)
            valid_entropy = self.y_valid_one_hot * np.log(y_valid_proba +epsilon)
            valid_cross_entropy = -np.sum(valid_entropy , axis = 1).mean()
            self.valid_loss_history.append(valid_cross_entropy)
            

            if self.best_losses > valid_cross_entropy :
                self.best_losses = valid_cross_entropy
                self.best_theta = self.Theta
                self.best_epoch = epoch

            if epoch %1000 == 0 : 
                print(f"Epoque : {epoch}, entropie coisé sur valid :{valid_cross_entropy} ")

            error = y_proba - self.y_train_one_hot 
            gradient = 1/ m * self.X_train.T @ error 

            self.Theta = self.Theta - eta * gradient
      
        self.best_theta = self.Theta
        print(f"\n\nModel final : \n\n->Parrametres du model : {self.best_theta} \n\n->Meilleur Xentropy : {self.best_losses } \n\n->Epoque de fin : {self.best_epoch}\n")

    def validation(self):
        logits = self.X_valid @ self.best_theta
        y_proba = self.softmax(logits)
        y_predict = y_proba.argmax(axis= 1)

        score = (y_predict == self.y_valid).mean()
        
        print(f"\n->Score de validation : {score}\n")

    def predict(self, petal_lenght, petal_width): 
        X=  [[1. , petal_lenght , petal_width]] 
        logits = X @ self.best_losses
        y_proba = self.softmax(logits)
        y_predict = y_proba.argmax(axis=1)

        



    def plot_loss(self):
        """Trace la courbe de perte pour l'entraînement et la validation."""
        plt.figure(1,figsize=(10, 6))
        plt.scatter(self.best_epoch,self.best_losses, c = 'red' , marker = 'o',label= "Best model")
        plt.plot(self.train_loss_history, label="Perte d'entraînement")
        plt.plot(self.valid_loss_history, label="Perte de validation")
        plt.xlabel("Époque")
        plt.ylabel("Entropie croisée")
        plt.title("Évolution de la perte pendant l'entraînement")
        
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_decision_boundary(self):
        plt.figure(2,figsize=(10, 6))
        """Trace les frontières de décision du modèle."""
        # Créer une grille pour les frontières de décision
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

        # Standardiser les points de la grille
        grid = np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()]
        grid[:, 1:] = (grid[:, 1:] - self.mean) / self.std

        # Prédire les probabilités pour chaque point de la grille
        logits = grid @ self.Theta
        y_proba = self.softmax(logits)
        y_pred = np.argmax(y_proba, axis=1)

        # Remettre en forme pour le tracé
        y_pred = y_pred.reshape(xx.shape)

        # Tracer la frontière de décision

        plt.contourf(xx, yy, y_pred, cmap=plt.cm.brg, alpha=0.3)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=plt.cm.brg, edgecolors='k')
        plt.xlabel("Longueur des pétales (cm)")
        plt.ylabel("Largeur des pétales (cm)")
        plt.title("Frontières de décision du modèle Softmax")
        plt.grid(True)
        plt.show()


def main() :
    
    model = SoftmaxRegressor()
    model.load_data()
    model.preprocess()
    model.grad_desc_go_back()
    model.validation() 
    model.plot_loss()  # Tracer la courbe de perte
    model.plot_decision_boundary()  # Tracer les frontières de décision
    
if __name__ == "__main__":
    main()



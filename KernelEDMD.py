import numpy as np
import scipy as sp
from scipy.spatial import distance
np.random.seed(3633914)
class kEDMD:
    def __init__(self, X, Y, kernel="gaussian", **kwargs):
        """
        X: N_dim * M_data
        Y: N_dim * M_data
        """
        self.X = X
        self.Y = Y
        self.kernel = kernel
        
        if self.kernel == "gaussian":
            assert "eps" in kwargs, "gaussian kernel needs eps param"
            self.eps = kwargs["eps"]
            self.f = self.gramian
        elif self.kernel == "polynomial":
            assert ("p" in kwargs) and ("c" in kwargs), "polynomial kernel needs c and p param"
            self.p = kwargs["p"]
            self.c = kwargs["c"]
            if "gamma" in kwargs:
                self.gamma = kwargs["gamma"]
            else:
                self.gamma = 1.0
            self.f = self.polynomial
        
    def fit(self, truncation=1e-3):
        
        self.G = self.f(self.X, self.X)
        self.A = self.f(self.X, self.Y)
        
        # 低ランク近似
        self.U, self.s, self.V = sp.linalg.svd(self.G)
        mask = self.s > truncation
        self.U = self.U[:, mask]
        self.V = self.V[mask, :]
        self.s = self.s[mask]
        S_inv = np.diag(1/self.s)
        # カーネルクープマン行列を求める
        self.K = self.V.T @ S_inv @ self.U.T @ self.A
        self.calc_modes(truncation=truncation)
        return None
        
    def calc_modes(self, truncation=1e-3):
        # 固有値,固有ベクトルを求める
        self.eigenvalues, self.eigenvectors = sp.linalg.eig(self.K)

        # Kを求めるときに使った各データ点x1, x2, ..., xMを用いて，固有関数の値を求める
        # Gは対称行列なので下のコードのように転置を取らなくても良い
        # 低ランク近似（必要に応じて、必要なければtruncation=0にすれば良い）
        self.modes = sp.linalg.pinv(self.G.T @ self.eigenvectors, rtol=truncation) @ self.X.T
        return None

    def calc_eigenfunction(self, x):
        M = self.X.shape[1] # num_data
        self.eigenfuncion = np.zeros([M])
        
        #次の座標を予測したいデータ点を入力とする固有関数の値を求める
        if self.kernel == "gaussian":
            x = x.reshape(-1,1)
            phi_Xx = np.exp(-np.linalg.norm(self.X - x, axis=0)**2/(2*(self.eps**2))) @ self.eigenvectors
        elif self.kernel == "polynomial":
            phi_Xx = (self.c + x.T @ self.X)**self.p @ self.eigenvectors
        return phi_Xx
        
    def predict(self, x):
        """
        x: 1 * N_dim
        return: 1 * N_dim
        """
        phi_Xx = self.calc_eigenfunction(x)
        return np.real((self.eigenvalues * phi_Xx.flatten()) @ self.modes).T

    def gramian(self, X, Y):
        return np.exp(-distance.cdist(X.T, Y.T, 'sqeuclidean')/(2*(self.eps**2)))
         
    def polynomial(self, X, Y):
        """
        c: param
        d: order
        X: D * M次元
        Y: D * M次元
        return Gram matrix, each element = k(x_i, y_i)
        """
        M = X.shape[1]
        G = np.zeros([M, M])
        for i in range(M):
            for j in range(M):
                G[i, j] += (self.gamma*(Y[:, i].T @X[:, j]) + self.c)**self.p
        return G
        
if __name__ == "__main__":
    X = np.array([[1, 1, 3],
                 [1, 2, 5]])
    Y = np.array([[1, 3, 4],
                 [2, 5, 6]])
    x = np.array([[1], 
                  [1]])
    #kedmd = kEDMD(X, Y, kernel="polynominal", c=0.1, p=5)
    kedmd = kEDMD(X, Y, kernel="gaussian", eps=2)
    kedmd.fit()
    print(kedmd.predict(x))
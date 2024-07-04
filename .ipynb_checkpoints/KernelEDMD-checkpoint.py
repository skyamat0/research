import numpy as np
import scipy as sp
from scipy.spatial import distance
np.random.seed(3633914)
class OnlineKernelEDMD:
    def __init__(self, X, Y, kernel="gaussian", **kwargs):
        """
        X: N_dim * M_data
        Y: N_dim * M_data
        """
        # For updating, this class needs to preserve previous data.
        self.X = X
        self.Y = Y
        self.kernel = kernel
        
        if self.kernel == "gaussian":
            assert "eps" in kwargs, "gaussian kernel needs eps param"
            self.eps = kwargs["eps"]
            self.f = self.gaussian
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
        self.s = self.s[mask]
        self.U = self.U[:, mask]
        self.V = self.V[mask, :]
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
        """
        x: N_dim * 1
        """
        M = self.X.shape[1] # num_data
        self.eigenfuncion = np.zeros([M])
        
        #次の座標を予測したいデータ点を入力とする固有関数の値を求める
        if self.kernel == "gaussian":
            phi_Xx = np.exp(-np.linalg.norm(self.X - x.reshape(-1,1), axis=0)**2/(2*(self.eps**2))) @ self.eigenvectors
        elif self.kernel == "polynomial":
            phi_Xx = (self.c + x.T @ self.X)**self.p @ self.eigenvectors
        return phi_Xx
        
    def predict(self, x):
        """
        x: N_dim * 1
        return: N_dim * 1
        """
        phi_Xx = self.calc_eigenfunction(x)
        return np.real((self.eigenvalues * phi_Xx.flatten()) @ self.modes)

    def gaussian(self, X, Y):
        """
        X: N_dim * M_data
        Y: N_dim * M_data
        return Gram matrix, ij element = k(x_j, y_i)
        """
        M = X.shape[1]
        G = np.zeros([M, M])
        for i in range(M):
            for j in range(M):
                G[i, j] += np.exp(-np.linalg.norm(Y[:, i] - X[:, j])**2/(2*(np.sqrt(2)**2)))
        return G
    def polynomial(self, X, Y):
        """
        c: param
        d: order
        X: N_dim * M_data
        Y: N_dim * M_data
        return Gram matrix, each element = k(x_i, y_i)
        """
        M = X.shape[1]
        G = np.zeros([M, M])
        for i in range(M):
            for j in range(M):
                G[i, j] += (self.gamma*(Y[:, i].T @X[:, j]) + self.c)**self.p
        return G
    @classmethod
    def svd_update(self, U, s, V, a, b, w=1):
        """
        a: col vector
        b: col vector
        """
        m = U.T @ a
        n = V @ b
        p = a - U @ m
        q = b - V.T @ n
        Ra = np.linalg.norm(p)
        Rb = np.linalg.norm(q)
        
        P = p/Ra
        Q = q/Rb
        
        K = np.append(m, Ra*w).reshape(-1,1) @ np.append(n, Rb*w).reshape(-1,1).T
        K[:-1, :-1] += np.diag(s)
        U_prime, s_prime, V_prime = np.linalg.svd(K)
        UpdatedU = np.hstack([U, P]) @ U_prime
        UpdatedV = np.hstack([V.T, Q]) @ V_prime.T
        
        return UpdatedU, s_prime, UpdatedV.T, U, P, U_prime

    def update(self, new_x, new_y, swap_index=0, truncation=1e-3, w=1, greedy=False):
        """
        Description:
            swap old data and new data from gramian G and covariance matrix A
            If swap_index = 0, swap oldest data. In other words, swap_index=0 means Sliding Window Update.
        Args:
            new_x, new_y: column vector(N_dim * 1)
        Return:
            Kernel Koopman Matrix
        """
        # swap old data and new data from Data matrix
        self.X = np.delete(self.X, swap_index, 1)
        self.X = np.insert(self.X, swap_index, new_x, 1)
        self.Y = np.delete(self.Y, swap_index, 1)
        self.Y = np.insert(self.Y, swap_index, new_y, 1)
        # calc kernel vector, k_ij = k(x_i, x_new)
        if self.kernel == "gaussian":
            d = np.exp(-np.linalg.norm(self.X- new_x.reshape(-1,1), axis=0)**2/(2*(self.eps**2)))
            self.A[:, swap_index] = np.exp(-np.linalg.norm(self.Y - new_x.reshape(-1,1), axis=0)**2/(2*(self.eps**2)))
            self.A[swap_index, :] = np.exp(-np.linalg.norm(self.X - new_y.reshape(-1,1), axis=0)**2/(2*(self.eps**2)))
            
        elif self.kernel == "polynomial":
            d = (self.c + new_x.reshape(1, -1) @ self.X)**self.p # x_2 ~ x_M+1
            self.A[:, swap_index] = (self.c + new_x.reshape(1, -1) @ self.Y)**self.p
            self.A[swap_index, :] = (self.c + new_y.reshape(1, -1) @ self.X)**self.p
            
        # Gの任意のデータを新しいデータにlow rank based updateする
        c = self.G[:, swap_index].reshape(-1,1)
        d = d.reshape(-1,1)
        b = np.zeros(self.V.shape[1]).reshape(-1,1)
        b[swap_index] += 1
        _U, _s, _V , U1,P1,U_prime1= self.svd_update(self.U, self.s, self.V, d-c, b, w)
        if greedy == True:
            _s = _s[:-1]
            _U = _U[:, :-1]
            _V = _V[:-1, :]
        else:
            mask = _s > truncation
            _s = _s[mask]
            _U = _U[:,mask]
            _V = _V[mask,:]
        
        # swap後の行列を転置して，再度更新
        c[swap_index] = d[swap_index]
        self.U, self.s, self.V, U2,P2,U_prime2 = self.svd_update(_V.T, _s, _U.T, d-c, b, w)

        # 求めたsを低ランク近似
        if greedy == True:
            self.s = self.s[:-1]
            self.U = self.U[:, :-1]
            self.V = self.V[:-1, :]
        else:
            mask = self.s > truncation
            self.s = self.s[mask]
            self.U = self.U[:, mask]
            self.V = self.V[mask, :]
        
        # 低ランク近似したものをGとする
        # self.s *= 0.95
        self.G = self.U @ np.diag(self.s) @ self.V
        
        # Gの逆行列を計算して新しいカーネルKoopman行列を求める
        S_inv = np.diag(1/self.s)
        G_inv = self.V.T @ S_inv @ self.U.T
        self.K = G_inv @ self.A
        return self.K, U1, P1, U_prime1, U2, P2, U_prime2
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
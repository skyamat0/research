import numpy as np
import scipy as sp
from scipy.spatial import distance

def svd_update(U, s, V, a, b):
    """
    kernelEDMDの場合は
    1. rollして，一番古い行・列を一番下・右に持ってくる
    2. swapupdate を適用
    3. 転置して適用
    で，一番古いものが新しいものに更新される．
    """
    m = U.T @ a
    n = V @ b
    p = a - U @ m
    q = b - V.T @ n
    Ra = np.linalg.norm(p)
    Rb = np.linalg.norm(q)
    P = p/Ra
    Q = q/Rb
    
    K = np.append(m, Ra).reshape(-1,1) @ np.append(n, Rb).reshape(-1,1).T
    K[:-1, :-1] += np.diag(s)
    U_prime, s_prime, V_prime = np.linalg.svd(K)
    UpdatedU = np.hstack([U, P]) @ U_prime
    UpdatedV = np.hstack([V.T, Q]) @ V_prime.T
    
    return UpdatedU, s_prime, UpdatedV.T

def gramian(X, sigma=np.sqrt(2)):
    return np.exp(-distance.cdist(X.T, X.T, 'sqeuclidean')/(2*(sigma**2)))
def gaussian_kernel(X, x, sigma=np.sqrt(2)):
    _X = np.vstack([X, x])
    new_x = _X - x
    return np.exp((-np.linalg.norm(new_x, axis=1)**2) / (2*(sigma**2)))[1:]

if __name__ == "__main__":
    X = np.array([[1, 2],
                 [3, 5],
                 [7, 6],
                 [9, 10]])
    x = np.array([[5, 7]])
    G = gramian(X.T)
    print(G)
    G = np.roll(G, shift=G.shape[0]-1, axis=(0,1))
    print("=====")
    print("rolled")
    print(G)
    print("=====")
    
    # 消すやつ [G_X c]
    c = G[:, -1].reshape(-1,1) 
    # 追加するやつ [G_X d]
    d = gaussian_kernel(X, x).reshape(-1,1)
    U, s, V = np.linalg.svd(G, full_matrices=False) # 3*2
    
    # G + a@b.Tで一番右の列がswapされる
    a = d - c
    b = np.zeros(V.shape[0]).reshape(-1,1)
    b[-1] += 1
    
    # ランクrに近似
    print(s)
    mask = s >= s.max()
    s = s[mask]
    U = U[:, mask]
    V = V[mask, :]
    print("V:", V.shape)
    print("b:", b.shape)
    U, s, V = svd_update(U, s, V, a, b)
    print("before")
    print(G)
    print("update by")
    print(d)
    print("after")
    print(U @ np.diag(s) @ V)
    print("======")
    print("before")
    print((U @ np.diag(s) @ V).T)
    U, s, V = svd_update(V.T, s, U.T, a, b)
    print("update by")
    print(d)
    print("after")
    print(U @ np.diag(s) @ V)
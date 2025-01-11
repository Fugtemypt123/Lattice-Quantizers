import numpy as np
from numpy.linalg import det
from scipy.linalg import block_diag

import os
from tqdm import tqdm

from train import *

# fix random seed for reproducibility
np.random.seed(0)

def volume_of_lattice(B):
    """
    计算生成矩阵 B 的体积 = |det(B)|。
    """
    return abs(det(B))

def compute_Ei_from_Gi(Vi, Gi, ni):
    """
    从 低维晶格 i 的 NSM = Gi,
       体积 = Vi,
       维度 = ni
    推出均方误差 E_i:
        E_i = n_i * G_i * (V_i)^(2 / n_i).
    这是因为 G_i = E_i / [n_i * (V_i^(2/n_i))].
    """
    Ei = ni * Gi * (Vi**(2.0/ni))
    return Ei

def compute_nsm_product_lattice(E_list, V_list, dim_list, a_list):
    """
    根据论文中的公式计算乘积晶格的NSM:
       G(a) = E(a) / [ n * (V(a))^(2/n) ],
    其中:
       E(a) = sum_i [ a_i^2 * E_i ],
       V(a) = Π_i [ a_i^(n_i) * V_i ],
       n = sum_i n_i.
    """
    n = sum(dim_list)  # 总维度

    # 计算 E(a)
    E_val = 0.0
    for Ei, ai in zip(E_list, a_list):
        E_val += (ai**2) * Ei

    # 计算 V(a)
    V_val = 1.0
    for Vi, ni, ai in zip(V_list, dim_list, a_list):
        V_val *= (ai**ni) * Vi

    # 计算 NSM
    G_val = E_val / (n * (V_val**(2.0/n)))
    return G_val

def find_optimal_scales(V_list, G_list, dim_list):
    """
    根据论文第四节(例如Theorem 4)，
    在子晶格体积 Vi、NSM Gi、维度 ni 已知的情况下，
    求使乘积晶格NSM最小的缩放因子 a_i。
    
    核心关系:
       a_i^2 * Vi^(2/ni) * Gi = a_j^2 * Vj^(2/nj) * Gj,  对所有 i,j
    并施加一个规范化条件，如  Π_i [a_i^(n_i)] = 1.
    
    此处的实现思路:
     1) 令 a_0 = 1 作为基准；
     2) 按照 a_i = sqrt( (V0^(2/n0)*G0)/(Vi^(2/ni)*Gi ) ) 得到比值；
     3) 最后统一乘以 alpha 保证乘积=1。
    """
    k = len(V_list)
    # 1) 选 i=0 作为参考
    n0 = dim_list[0]
    base_val = (V_list[0]**(2.0/n0)) * G_list[0]  # V0^(2/n0)*G0
    
    ratio_list = [1.0]*k
    for i in range(1, k):
        ni = dim_list[i]
        val_i = (V_list[i]**(2.0/ni))*G_list[i]
        ratio_list[i] = np.sqrt(base_val / val_i)  # a_i / a_0

    # 2) 此时 a_0=1, a_i= ratio_list[i]
    #    要满足 Π_i (a_i^(n_i))=1 => alpha^n * Π_i(...)=1
    n_total = sum(dim_list)
    product_term = 1.0
    for i in range(k):
        product_term *= ratio_list[i]**(dim_list[i])
    alpha = (1.0/product_term)**(1.0/n_total)

    # 3) 得到 a_opt
    a_opt = [alpha * r for r in ratio_list]
    return a_opt

def construct_product_lattice(B_list, a_list):
    """
    给定 B_list=[B1,B2,...,Bk]以及对应缩放a_list=[a1,a2,...,ak],
    返回高维'块对角'的生成矩阵 block_diag(a1 B1, a2 B2, ..., ak Bk).
    """
    scaled = [a_i * B_i for (B_i, a_i) in zip(B_list, a_list)]
    return block_diag(*scaled)


# ======================== 示例演示 =================================

if __name__ == "__main__":
    # for i in range(2, 13):
    #     B = np.load(f'./lattice_B_{i}/final.npy')
    #     np.save(f'./lattice_B_{i}/best.npy', B)
    # raise ValueError('with great power comes great responsibility')
    # 假设我们有k=2个低维晶格:
    #   Λ1: 2维, 生成矩阵 B1, 体积V1, NSM=G1
    #   Λ2: 1维, 生成矩阵 B2, 体积V2, NSM=G2
    # 这里只是示例，参数可能是随意的。

    # B1 = np.array([[1.0, 0.0],
    #                [0.0, 1.0]])  # 2D
    # B2 = np.array([[1.0]])       # 1D

    for jj in tqdm(range(64, 65)):  # 只使用两个32合成64

        n = jj
        best_G = 10000
        best_n1 = 0
        best_n2 = 0

        for i in range(32, 33):  # 只使用两个32合成64
            n1 = i
            n2 = n - i
            B1 = np.load(f'./lattice_B_{n1}/best.npy')
            B2 = np.load(f'./lattice_B_{n2}/best.npy')

            V1 = volume_of_lattice(B1)  # 2D体积
            V2 = volume_of_lattice(B2)  # 1D'体积'即|1.0|=1

            # 这里假设我们已知 (或通过其它手段计算得到) Λ1与Λ2的NSM:
            # G1 = 0.080187537       # 示例
            # G2 = 0.083333333       # 示例
            if os.path.exists(f"./lattice_B_{n1}/best_G.txt"):
                with open(f"./lattice_B_{n1}/best_G.txt", 'r') as f:
                    G1 = float(f.read())
            else:
                G1 = compute_nsm(B1, iters=20000)
                with open(f"./lattice_B_{n1}/best_G.txt", 'w') as f:
                    f.write(str(G1))
            if os.path.exists(f"./lattice_B_{n2}/best_G.txt"):
                with open(f"./lattice_B_{n2}/best_G.txt", 'r') as f:
                    G2 = float(f.read())
            else:
                G2 = compute_nsm(B2, iters=20000)
                with open(f"./lattice_B_{n2}/best_G.txt", 'w') as f:
                    f.write(str(G2))

            # print("=== 子晶格信息 ===")
            # print(f"  B1=\n{B1},  V1={V1:.4f},  G1={G1}")
            # print(f"  B2=\n{B2},  V2={V2:.4f},  G2={G2}")

            # ========== 1) 合成高维晶格的NSM ==========

            # a) 先算各子晶格的 E_i
            E1 = compute_Ei_from_Gi(V1, G1, n1)
            E2 = compute_Ei_from_Gi(V2, G2, n2)

            # b) 找到最优缩放因子 a_opt
            #    (令 乘积晶格 G 最小)
            dim_list = [n1, n2]
            V_list   = [V1, V2]
            G_list   = [G1, G2]
            a_opt = find_optimal_scales(V_list, G_list, dim_list)

            # print("\n---> 最优缩放因子 a_opt =", a_opt)

            # c) 拼成高维生成矩阵
            B_prod = construct_product_lattice([B1,B2], a_opt)
            # print("合成后高维 生成矩阵 B_prod =\n", B_prod)

            # d) 用公式法(低维合成)得到 高维NSM
            # E_list = [E1, E2]
            # G_val_formula = compute_nsm_product_lattice(E_list, V_list, dim_list, a_opt)
            # print(f"高维乘积晶格(公式法) 的 NSM = {G_val_formula:.6f}")

            # ========== 2) 直接采样的方式估计 高维NSM ==========

            G_val_sampling = compute_nsm(B_prod, iters=500)
            # print(f"高维乘积晶格(采样法) 的 NSM ~= {G_val_sampling:.6f}")

            # print("\n对比：\n  公式法(基于低维合成) = {:.6f}\n  采样法(近似)       = {:.6f}"
            #     .format(G_val_formula, G_val_sampling))
            
            if G_val_sampling < best_G:
                best_G = G_val_sampling
                best_B = B_prod
                best_n1 = n1
                best_n2 = n2

        # print(f"=================== optimized lattice ====================")
        # print(f"best_B = {best_B}")
        print(f'n = {n}')
        print(f"best_G = {best_G}")
        print(f"best_n1 = {best_n1}")
        print(f"best_n2 = {best_n2}")

        os.makedirs(f"./lattice_B_{n}", exist_ok=True)

        if os.path.exists(f"./lattice_B_{n}/final.npy"):
            B_final = np.load(f"./lattice_B_{n}/final.npy")
            if os.path.exists(f"./lattice_B_{n}/best_G.txt"):
                with open(f"./lattice_B_{n}/best_G.txt", 'r') as f:
                    G_final = float(f.read())
            else:
                G_final = compute_nsm(B_final, iters=500)
                with open(f"./lattice_B_{n}/best_G.txt", 'w') as f:
                    f.write(str(G_final))
            if best_G < G_final:
                print(f"best_G = {best_G} < G_final = {G_final}")
                np.save(f"./lattice_B_{n}/best.npy", best_B)
                # 保存best_G这个浮点数
                with open(f"./lattice_B_{n}/best_G.txt", 'w') as f:
                    f.write(str(best_G))
        else:
            np.save(f"./lattice_B_{n}/best.npy", best_B)
            # 保存best_G这个浮点数
            with open(f"./lattice_B_{n}/best_G.txt", 'w') as f:
                f.write(str(best_G))
    







# ==================== optimized lattice ====================

# B_10 = [[ 1.33267874  0.          0.          0.          0.          0.
#    0.          0.          0.          0.        ]
#  [-0.51677576  1.29032294  0.          0.          0.          0.
#    0.          0.          0.          0.        ]
#  [ 0.24860906  0.71746374  1.25587127  0.          0.          0.
#    0.          0.          0.          0.        ]
#  [-0.14623105  0.25527198  0.20810982  1.19217875  0.          0.
#    0.          0.          0.          0.        ]
#  [-0.2676491  -0.66321142  0.4538925   0.25283892  0.95563584  0.
#    0.          0.          0.          0.        ]
#  [ 0.16786354 -0.01054421 -0.09594673  1.02195047  0.03638536  1.09497572
#    0.          0.          0.          0.        ]
#  [-0.50673155 -0.05396517  0.4477078   0.42368687  0.18029331  0.57344126
#    0.63004389  0.          0.          0.        ]
#  [ 0.42773975  0.06079724  0.36930222 -0.05250003 -0.23824401  0.09555951
#   -0.2187094   0.86852056  0.          0.        ]
#  [ 0.05689535  0.0987014   0.6033408   0.31390286 -0.52303104  0.26431479
#    0.45198169 -0.10037699  0.90282694  0.        ]
#  [ 0.10217592 -0.51938943  0.26854412  0.5117036  -0.43231376  0.09275387
#   -0.08184687  0.22888571  0.44744628  0.71316176]]
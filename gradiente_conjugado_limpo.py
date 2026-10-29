"""
Método das Direções Conjugadas para Otimização Irrestrita
=========================================================
Relatório Final PIBIC (PIB-E/0074/2020) — UFAM / FAPEAM (2021)
Autor: Francisco Lima da Silva
Orientador: Prof. Dr. Roberto Cristóvão Mesquita Silva

Este módulo implementa de forma limpa, coesa e didática todos os algoritmos
descritos no Capítulo 3 do relatório:
1. Busca Exata pela Seção Áurea em 2 fases (Algoritmo 2 - Seção 3.2.1)
2. Busca Inexata pela Condição de Armijo (Algoritmo 3 - Seção 3.2.2)
3. Método das Direções Conjugadas para Funções Quadráticas (Seção 3.3.2)
4. Extensão para Funções Não-Quadráticas de 3º e 4º Grau com variantes de
   Fletcher-Reeves (FR) e Polak-Ribière (PR) (Seção 3.3.3 e Seção 3.4)
"""

from typing import Callable, Literal
import numpy as np


# =============================================================================
# 1. FUNÇÕES BASE (QUADRÁTICAS)
# =============================================================================

def funcao_quadratica(x: np.ndarray, A: np.ndarray, b: np.ndarray, c: float = 0.0) -> float:
    """Calcula o valor escalar de f(x) = (1/2) * x^T A x + b^T x + c."""
    return float(0.5 * (x @ A @ x) + (b @ x) + c)


def gradiente(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calcula o vetor gradiente ∇f(x) = A x + b de uma função quadrática."""
    return A @ x + b


# =============================================================================
# 2. MÉTODOS DE BUSCA UNIDIRECIONAL (SEÇÃO 3.2 DO RELATÓRIO)
# =============================================================================

def busca_secao_aurea(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    d: np.ndarray,
    eps: float = 1e-7,
    rho: float = 0.1,
    max_iter: int = 500,
) -> float:
    """
    Algoritmo 2 (Seção 3.2.1): Busca Exata pelo Método da Seção Áurea em 2 fases
    para minimizar φ(t) = f(x + t * d) com t > 0.

     Fase 1: Encontra um intervalo [a, b] que contém um minimizador local de φ(t).
     Fase 2: Reduz o intervalo [a, b] usando a razão áurea (θ1 ≈ 0.382, θ2 ≈ 0.618)
             até que (b - a) <= eps.
    """
    phi = lambda t: float(f(x + t * d))

    theta1 = (3.0 - np.sqrt(5.0)) / 2.0  # ≈ 0.381966
    theta2 = 1.0 - theta1                # ≈ 0.618034

    # Fase 1: Obtenção do intervalo [a, b]
    a = 0.0
    s = rho
    b = 2.0 * rho

    iter_fase1 = 0
    while phi(b) < phi(s) and iter_fase1 < max_iter:
        a = s
        s = b
        b = 2.0 * b
        iter_fase1 += 1

    # Fase 2: Redução do intervalo [a, b] pela razão áurea
    u = a + theta1 * (b - a)
    v = a + theta2 * (b - a)
    phi_u = phi(u)
    phi_v = phi(v)

    while (b - a) > eps:
        if phi_u < phi_v:
            b = v
            v = u
            phi_v = phi_u
            u = a + theta1 * (b - a)
            phi_u = phi(u)
        else:
            a = u
            u = v
            phi_u = phi_v
            v = a + theta2 * (b - a)
            phi_v = phi(v)

    return 0.5 * (u + v)


def busca_armijo(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    d: np.ndarray,
    eta: float = 1e-4,
    gamma: float = 0.5,
    t_inicial: float = 1.0,
) -> float:
    """
    Algoritmo 3 (Seção 3.2.2): Busca Inexata pela Condição de Armijo.
    Encontra t = gamma^k * t_inicial tal que:
        f(x + t * d) <= f(x) + eta * t * (∇f(x)^T d)
    """
    t = t_inicial
    fx = float(f(x))
    derivada_direcional = float(grad_f(x) @ d)

    # Garante que d seja direção de descida
    if derivada_direcional >= 0:
        d = -grad_f(x)
        derivada_direcional = float(grad_f(x) @ d)

    while float(f(x + t * d)) > fx + eta * t * derivada_direcional:
        t *= gamma
        if t < 1e-14:
            break

    return t


# =============================================================================
# 3. DIREÇÕES CONJUGADAS PARA FUNÇÕES QUADRÁTICAS (SEÇÃO 3.3.2)
# =============================================================================

def gradiente_conjugado(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 100,
    verbose: bool = True,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Algoritmo das Direções Conjugadas para Funções Quadráticas (2º Grau):
        f(x) = (1/2) * x^T A x + b^T x + c
    Converge em no máximo n passos quando A é simétrica e definida positiva.
    """
    x = np.atleast_1d(x0).astype(float).copy()
    A = np.atleast_2d(A).astype(float)
    b = np.atleast_1d(b).astype(float)
    historico = [x.copy()]

    g = gradiente(x, A, b)
    d = -g

    for k in range(max_iter):
        norma_g = float(np.linalg.norm(g))
        if verbose:
            print(f"Iteração {k:2d} | x = {np.round(x, 6)} | ||grad|| = {norma_g:.2e}")

        if norma_g < tol:
            if verbose:
                print(f"-> Convergência atingida na iteração {k}!\n")
            break

        Ad = A @ d
        curvatura = float(d @ Ad)

        # Passo ótimo exato (Equação 3.11 do relatório)
        t = -float(g @ d) / curvatura

        # Atualização do ponto (Equação 3.9)
        x = x + t * d
        historico.append(x.copy())

        # Novo gradiente
        g_novo = gradiente(x, A, b)

        # Coeficiente Beta (Equação 3.13)
        beta = float(Ad @ g_novo) / curvatura

        # Nova direção A-conjugada
        d = -g_novo + beta * d
        g = g_novo

    return x, historico


# =============================================================================
# 4. EXTENSÃO PARA FUNÇÕES NÃO-QUADRÁTICAS — 3º E 4º GRAU (SEÇÃO 3.3.3)
# =============================================================================

def gradiente_conjugado_nao_linear(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    metodo_beta: Literal["FR", "PR"] = "FR",
    metodo_busca: Literal["secao_aurea", "armijo"] = "secao_aurea",
    tol: float = 1e-6,
    max_iter: int = 500,
    verbose: bool = True,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Algoritmo de Direções Conjugadas para Funções Não-Quadráticas (Seção 3.3.3).

    Parâmetros:
        f            : Função objetivo escalar f(x).
        grad_f       : Função que retorna o vetor gradiente ∇f(x).
        x0           : Ponto inicial.
        metodo_beta  : 'FR' (Fletcher-Reeves, Eq. 3.16) ou 'PR' (Polak-Ribière, Eq. 3.15).
        metodo_busca : 'secao_aurea' (busca exata) ou 'armijo' (busca inexata).
        tol          : Tolerância para a norma do gradiente ||∇f(x)||.
        max_iter     : Máximo de iterações.
    """
    x = np.atleast_1d(x0).astype(float).copy()
    n = len(x)
    historico = [x.copy()]

    g = np.atleast_1d(grad_f(x)).astype(float)
    d = -g

    for k in range(max_iter):
        norma_g = float(np.linalg.norm(g))
        if verbose and (k < 10 or k % 10 == 0):
            print(f"Iteração {k:3d} | x = {np.round(x, 6)} | ||grad|| = {norma_g:.2e}")

        if norma_g < tol:
            if verbose:
                print(f"-> Convergência ({metodo_beta} + {metodo_busca}) em {k} iterações!\n")
            break

        # Garante que d seja direção de descida; se não for, reinicia com -g
        if float(g @ d) >= 0:
            d = -g

        # 1. Calcula o tamanho do passo t_k (Seção Áurea ou Armijo)
        if metodo_busca == "secao_aurea":
            t = busca_secao_aurea(f, x, d)
        else:
            t = busca_armijo(f, grad_f, x, d)

        # 2. Atualiza x^{k+1} = x^k + t_k * d^k
        x = x + t * d
        historico.append(x.copy())

        # 3. Novo gradiente g_{k+1}
        g_novo = np.atleast_1d(grad_f(x)).astype(float)

        # 4. Reinicialização periódica a cada n passos: (k + 1) mod n == 0
        if (k + 1) % n == 0:
            beta = 0.0
        else:
            denom = float(g @ g)
            if metodo_beta == "FR":
                # Fletcher-Reeves (Equação 3.16)
                beta = float(g_novo @ g_novo) / denom
            else:
                # Polak-Ribière (Equação 3.15)
                beta = float(g_novo @ (g_novo - g)) / denom

        # 5. Nova direção conjugada
        d = -g_novo + beta * d
        g = g_novo

    return x, historico


def gradiente_conjugado_terceiro_grau(
    grad_f: Callable[[np.ndarray], np.ndarray],
    hess_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 100,
    verbose: bool = True,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Variante Newton-CG (usando a Hessiana local H(x)) para polinômios de 3º grau.
    """
    x = np.atleast_1d(x0).astype(float).copy()
    n = len(x)
    historico = [x.copy()]

    g = np.atleast_1d(grad_f(x)).astype(float)
    d = -g

    for k in range(max_iter):
        norma_g = float(np.linalg.norm(g))
        if verbose:
            print(f"Iteração {k:2d} | x = {np.round(x, 6)} | ||grad|| = {norma_g:.2e}")

        if norma_g < tol:
            if verbose:
                print(f"-> Convergência atingida na iteração {k}!\n")
            break

        H = np.atleast_2d(hess_f(x)).astype(float)
        curvatura = float(d @ H @ d)
        t = -float(g @ d) / curvatura

        x = x + t * d
        historico.append(x.copy())

        g_novo = np.atleast_1d(grad_f(x)).astype(float)
        beta = 0.0 if (k + 1) % n == 0 else float(g_novo @ g_novo) / float(g @ g)

        d = -g_novo + beta * d
        g = g_novo

    return x, historico


if __name__ == "__main__":
    print("=====================================================================")
    print(" REPRODUZINDO OS TESTES COMPUTACIONAIS DO RELATÓRIO PIBIC/UFAM (3.4)")
    print("=====================================================================\n")

    # 1. Quadrática 1D da Tabela 3.4.1 (A = 42.5, b = -12, x0 = 10 -> x* ≈ 0.28235294)
    print("--- 1. Quadrática 1D (Tabela 3.4.1: A = 42.5, b = -12, x0 = 10) ---")
    x_q1, _ = gradiente_conjugado(np.array([[42.5]]), np.array([-12.0]), np.array([10.0]))
    print(f"Resultado: {x_q1[0]:.16f} (Relatório: 0.2823529411764706)\n")

    # 2. Função de 4º Grau da Tabela 3.4.2: f(x) = x^4 - 4x^3 + 6x^2 - 4x + 1 = (x - 1)^4
    print("--- 2. Polinômio de 4º Grau (Tabela 3.4.2): f(x) = (x - 1)^4, x0 = 0 ---")
    f_4grau = lambda x: float((x[0] - 1.0) ** 4)
    g_4grau = lambda x: np.array([4.0 * (x[0] - 1.0) ** 3])
    x_4g, _ = gradiente_conjugado_nao_linear(f_4grau, g_4grau, np.array([0.0]), metodo_beta="FR")
    print(f"Mínimo encontrado: {x_4g[0]:.6f} (Exato: 1.0)\n")

    # 3. Função de 3º Grau da Tabela 3.4.2: f(x) = -x^3 + 4x^2 + 3x + 4, x0 = 0 -> x* = -1/3
    print("--- 3. Polinômio de 3º Grau (Tabela 3.4.2): f(x) = -x^3 + 4x^2 + 3x + 4, x0 = 0 ---")
    f_3grau_a = lambda x: float(-x[0]**3 + 4.0 * x[0]**2 + 3.0 * x[0] + 4.0)
    g_3grau_a = lambda x: np.array([-3.0 * x[0]**2 + 8.0 * x[0] + 3.0])
    x_3ga, _ = gradiente_conjugado_nao_linear(f_3grau_a, g_3grau_a, np.array([0.0]), metodo_beta="FR")
    print(f"Mínimo encontrado: {x_3ga[0]:.16f} (Relatório: -0.3333333747657136)\n")

    # 4. Função de 3º Grau da Tabela 3.4.2 (Inexata): f(x) = -x^3 + 6x^2 + 20x + 3, x0 = 0 -> x* ≈ -1.2660
    print("--- 4. Polinômio de 3º Grau (Tabela 3.4.2): f(x) = -x^3 + 6x^2 + 20x + 3, x0 = 0 ---")
    f_3grau_b = lambda x: float(-x[0]**3 + 6.0 * x[0]**2 + 20.0 * x[0] + 3.0)
    g_3grau_b = lambda x: np.array([-3.0 * x[0]**2 + 12.0 * x[0] + 20.0])
    x_3gb, _ = gradiente_conjugado_nao_linear(
        f_3grau_b, g_3grau_b, np.array([0.0]), metodo_beta="PR", metodo_busca="armijo"
    )
    print(f"Mínimo encontrado: {x_3gb[0]:.6f} (Newton no Relatório: -1.2660)\n")

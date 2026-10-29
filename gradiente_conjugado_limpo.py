"""
Método dos Gradientes Conjugados para Funções Quadráticas
=========================================================

Objetivo:
    Minimizar a função quadrática convexa:
        f(x) = (1/2) * x^T A x + b^T x + c

    onde:
        - A é uma matriz simétrica definida positiva (n x n)
        - b é um vetor (n,)
        - c é uma constante escalar
"""

import numpy as np


def funcao_quadratica(x: np.ndarray, A: np.ndarray, b: np.ndarray, c: float = 0.0) -> float:
    """Calcula o valor escalar de f(x) = (1/2) * x^T A x + b^T x + c."""
    return 0.5 * (x @ A @ x) + (b @ x) + c


def gradiente(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calcula o vetor gradiente ∇f(x) = A x + b."""
    return A @ x + b


def gradiente_conjugado(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Executa o Método dos Gradientes Conjugados.

    Parâmetros:
        A        : Matriz Hessiana simétrica definida positiva (n x n).
        b        : Vetor linear da função quadrática (n,).
        x0       : Ponto inicial de busca (n,).
        tol      : Tolerância para a norma do gradiente ||g_k|| (critério de parada).
        max_iter : Número máximo de iterações.

    Retorna:
        x_otimo   : Ponto de mínimo encontrado.
        historico : Lista com todos os pontos x_k visitados até a convergência.
    """
    x = x0.astype(float).copy()
    historico = [x.copy()]

    # Passo 0: Gradiente inicial e primeira direção de busca (oposta ao gradiente)
    g = gradiente(x, A, b)
    d = -g

    for k in range(max_iter):
        # Critério de parada: se o tamanho (norma) do gradiente for quase zero, chegamos ao mínimo
        norma_g = np.linalg.norm(g)
        print(f"Iteração {k:2d} | x = {np.round(x, 6)} | ||grad|| = {norma_g:.2e}")

        if norma_g < tol:
            print(f"\nConvergência atingida na iteração {k}!")
            break

        # Guarda o produto A @ d para não calculá-lo duas vezes na mesma iteração
        Ad = A @ d
        curvatura = d @ Ad  # d^T A d

        # Passo 1: Tamanho ótimo do passo (t_k) na direção d
        t = -(g @ d) / curvatura

        # Passo 2: Atualiza a posição (x_{k+1} = x_k + t_k * d_k)
        x = x + t * d
        historico.append(x.copy())

        # Passo 3: Calcula o novo gradiente no ponto atualizado
        g_novo = gradiente(x, A, b)

        # Passo 4: Coeficiente de conjugação (Beta_k) para manter A-ortogonalidade
        beta = (d @ A @ g_novo) / curvatura

        # Passo 5: Nova direção conjugada (d_{k+1} = -g_{k+1} + Beta_k * d_k)
        d = -g_novo + beta * d
        g = g_novo
    else:
        print("\nAviso: Número máximo de iterações atingido sem convergir.")

    return x, historico


if __name__ == "__main__":
    # =========================================================================
    # Exemplo 1: Os mesmos dados do código original do TCC (matriz diagonal)
    # =========================================================================
    print("=== EXEMPLO 1: Matriz Diagonal (Dados originais do TCC) ===")
    A_tcc = np.array([[15.0, 0.0], [0.0, 15.0]])
    b_tcc = np.array([8.0, 3.0])
    x0_tcc = np.array([10.0, 10.0])

    x_min1, _ = gradiente_conjugado(A_tcc, b_tcc, x0_tcc)
    print(f"Ponto de mínimo encontrado: {x_min1}")
    print(f"Solução exata (-A^-1 b)   : {-np.linalg.solve(A_tcc, b_tcc)}\n")

    # =========================================================================
    # Exemplo 2: Matriz com termos cruzados (converge em n = 2 passos!)
    # =========================================================================
    print("=== EXEMPLO 2: Matriz Simétrica Não-Diagonal (2 passos exatos) ===")
    A_geral = np.array([[4.0, 1.0], [1.0, 3.0]])
    b_geral = np.array([-1.0, -2.0])
    x0_geral = np.array([10.0, 10.0])

    x_min2, _ = gradiente_conjugado(A_geral, b_geral, x0_geral)
    print(f"Ponto de mínimo encontrado: {x_min2}")
    print(f"Solução exata (-A^-1 b)   : {-np.linalg.solve(A_geral, b_geral)}")

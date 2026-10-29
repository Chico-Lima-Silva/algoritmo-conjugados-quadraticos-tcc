"""
Método dos Gradientes Conjugados — 2º Grau (Quadráticas) e 3º Grau (Cúbicas)
============================================================================

Este módulo contém:
1. `gradiente_conjugado`: Para funções quadráticas (2º grau) em qualquer dimensão
   (ex.: 2x2 em R², 3x3 em R³), onde a matriz Hessiana A é constante.
2. `gradiente_conjugado_terceiro_grau`: Extensão não-linear (Fletcher-Reeves /
   Newton-CG) para polinômios de 3º grau, onde a curvatura (Hessiana H(x))
   varia a cada ponto x.
"""

from typing import Callable
import numpy as np


def funcao_quadratica(x: np.ndarray, A: np.ndarray, b: np.ndarray, c: float = 0.0) -> float:
    """Calcula o valor escalar de f(x) = (1/2) * x^T A x + b^T x + c."""
    return float(0.5 * (x @ A @ x) + (b @ x) + c)


def gradiente(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calcula o vetor gradiente ∇f(x) = A x + b de uma função de 2º grau."""
    return A @ x + b


def gradiente_conjugado(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 100,
    verbose: bool = True,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Executa o Método dos Gradientes Conjugados para funções quadráticas (2º grau).

    Parâmetros:
        A        : Matriz Hessiana simétrica definida positiva (n x n).
        b        : Vetor linear da função quadrática (n,).
        x0       : Ponto inicial de busca (n,).
        tol      : Tolerância para a norma do gradiente ||g_k|| (critério de parada).
        max_iter : Número máximo de iterações.
        verbose  : Se True, exibe o progresso a cada iteração.

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
        norma_g = float(np.linalg.norm(g))
        if verbose:
            print(f"Iteração {k:2d} | x = {np.round(x, 6)} | ||grad|| = {norma_g:.2e}")

        if norma_g < tol:
            if verbose:
                print(f"-> Convergência atingida na iteração {k}!\n")
            break

        # Guarda o produto A @ d para não calculá-lo duas vezes na mesma iteração
        Ad = A @ d
        curvatura = float(d @ Ad)  # d^T A d

        # Passo 1: Tamanho ótimo do passo (t_k) na direção d
        t = -float(g @ d) / curvatura

        # Passo 2: Atualiza a posição (x_{k+1} = x_k + t_k * d_k)
        x = x + t * d
        historico.append(x.copy())

        # Passo 3: Calcula o novo gradiente no ponto atualizado
        g_novo = gradiente(x, A, b)

        # Passo 4: Coeficiente de conjugação (Beta_k) para manter A-ortogonalidade
        beta = float(Ad @ g_novo) / curvatura

        # Passo 5: Nova direção conjugada (d_{k+1} = -g_{k+1} + Beta_k * d_k)
        d = -g_novo + beta * d
        g = g_novo
    else:
        if verbose:
            print("\nAviso: Número máximo de iterações atingido sem convergir.")

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
    Extensão do Gradiente Conjugado Não-Linear para polinômios de 3º grau (cúbicos)
    ou funções gerais duas vezes diferenciáveis.

    Diferença para o caso de 2º grau:
        - Em funções de 2º grau, a Hessiana A é constante e converge em até n passos.
        - Em funções de 3º grau, a Hessiana H(x) depende de x, então atualizamos a
          matriz de curvatura localmente a cada passo e usamos Fletcher-Reeves com
          reinício periódico a cada n iterações.
    """
    x = x0.astype(float).copy()
    n = len(x)
    historico = [x.copy()]

    g = grad_f(x)
    d = -g

    for k in range(max_iter):
        norma_g = float(np.linalg.norm(g))
        if verbose:
            print(f"Iteração {k:2d} | x = {np.round(x, 6)} | ||grad|| = {norma_g:.2e}")

        if norma_g < tol:
            if verbose:
                print(f"-> Convergência atingida na iteração {k}!\n")
            break

        # Em funções de 3º grau, a matriz Hessiana H(x) muda conforme x anda
        H = hess_f(x)
        curvatura = float(d @ H @ d)

        # Passo 1: Tamanho do passo usando a curvatura local (passo de Newton-CG)
        t = -float(g @ d) / curvatura

        # Passo 2: Atualiza o ponto x
        x = x + t * d
        historico.append(x.copy())

        # Passo 3: Novo gradiente
        g_novo = grad_f(x)

        # Passo 4: Fórmula de Fletcher-Reeves (com reinício a cada n passos)
        if (k + 1) % n == 0:
            beta = 0.0
        else:
            beta = float(g_novo @ g_novo) / float(g @ g)

        # Passo 5: Nova direção conjugada
        d = -g_novo + beta * d
        g = g_novo

    return x, historico


if __name__ == "__main__":
    # =========================================================================
    # TESTE 1: Função Quadrática (2º Grau) em 2D - Dados Originais do TCC
    # =========================================================================
    print("=== TESTE 1: Quadrática de 2º Grau (2x2 Diagonal - TCC) ===")
    A_2d_diag = np.array([[15.0, 0.0], [0.0, 15.0]])
    b_2d_diag = np.array([8.0, 3.0])
    x0_2d = np.array([10.0, 10.0])

    x_min1, _ = gradiente_conjugado(A_2d_diag, b_2d_diag, x0_2d)
    print(f"Mínimo encontrado: {x_min1}")
    print(f"Mínimo exato     : {-np.linalg.solve(A_2d_diag, b_2d_diag)}\n")

    # =========================================================================
    # TESTE 2: Função Quadrática (2º Grau) em 2D Acoplada (converge em 2 passos)
    # =========================================================================
    print("=== TESTE 2: Quadrática de 2º Grau (2x2 Simétrica Não-Diagonal) ===")
    A_2d = np.array([[4.0, 1.0], [1.0, 3.0]])
    b_2d = np.array([-1.0, -2.0])

    x_min2, _ = gradiente_conjugado(A_2d, b_2d, x0_2d)
    print(f"Mínimo encontrado: {x_min2}")
    print(f"Mínimo exato     : {-np.linalg.solve(A_2d, b_2d)}\n")

    # =========================================================================
    # TESTE 3: Função Quadrática em 3D (Matriz 3x3 - converge em 3 passos)
    # =========================================================================
    print("=== TESTE 3: Quadrática em 3 Variáveis (3x3 - Converge em 3 passos) ===")
    A_3d = np.array([
        [4.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0],
    ])
    b_3d = np.array([-5.0, -5.0, -3.0])  # Solução exata: [1, 1, 1]
    x0_3d = np.array([10.0, -5.0, 8.0])

    x_min3, _ = gradiente_conjugado(A_3d, b_3d, x0_3d)
    print(f"Mínimo encontrado: {x_min3}")
    print(f"Mínimo exato     : {-np.linalg.solve(A_3d, b_3d)}\n")

    # =========================================================================
    # TESTE 4: Função Polinomial de 3º Grau (Cúbica)
    # f(x1, x2) = (1/3)*(x1^3 + x2^3) + 2*x1^2 + 1.5*x2^2 - 5*x1 - 4*x2
    # Para x > 0:
    #   grad_f(x) = [x1^2 + 4*x1 - 5, x2^2 + 3*x2 - 4] -> zero em x* = [1.0, 1.0]
    #   hess_f(x) = [[2*x1 + 4, 0], [0, 2*x2 + 3]]
    # =========================================================================
    print("=== TESTE 4: Polinômio de 3º Grau (Cúbico) ===")

    def grad_cubico(x: np.ndarray) -> np.ndarray:
        return np.array([
            x[0]**2 + 4.0 * x[0] - 5.0,
            x[1]**2 + 3.0 * x[1] - 4.0,
        ])

    def hess_cubico(x: np.ndarray) -> np.ndarray:
        return np.array([
            [2.0 * x[0] + 4.0, 0.0],
            [0.0, 2.0 * x[1] + 3.0],
        ])

    x0_cubico = np.array([5.0, 5.0])
    x_min4, _ = gradiente_conjugado_terceiro_grau(grad_cubico, hess_cubico, x0_cubico)
    print(f"Mínimo encontrado: {x_min4}")
    print("Mínimo exato     : [1. 1.]")

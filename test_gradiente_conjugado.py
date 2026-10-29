"""
Testes Automatizados — Método das Direções Conjugadas (Relatório PIBIC/UFAM)
============================================================================

Para executar todos os testes no terminal:
    python -m unittest test_gradiente_conjugado.py -v
"""

import unittest
import numpy as np
from gradiente_conjugado_limpo import (
    gradiente_conjugado,
    gradiente_conjugado_nao_linear,
    gradiente_conjugado_terceiro_grau,
)


class TestGradienteConjugado(unittest.TestCase):
    """Bateria de testes para funções de 2º, 3º e 4º grau (incluindo a Seção 3.4 do Relatório)."""

    def test_segundo_grau_diagonal_tcc(self):
        """Teste 1: Função quadrática de 2º grau diagonal (código original de 2021)."""
        A = np.array([[15.0, 0.0], [0.0, 15.0]])
        b = np.array([8.0, 3.0])
        x0 = np.array([10.0, 10.0])

        x_otimo, historico = gradiente_conjugado(A, b, x0, verbose=False)
        esperado = np.array([-8.0 / 15.0, -3.0 / 15.0])

        np.testing.assert_allclose(x_otimo, esperado, rtol=1e-7, atol=1e-8)
        self.assertEqual(len(historico) - 1, 1)

    def test_segundo_grau_1d_tabela_relatorio(self):
        """Teste 2: Quadrática 1D da Tabela 3.4.1 do Relatório (A = 42.5, b = -12, x0 = 10)."""
        A = np.array([[42.5]])
        b = np.array([-12.0])
        x0 = np.array([10.0])

        x_otimo, _ = gradiente_conjugado(A, b, x0, verbose=False)
        self.assertAlmostEqual(float(x_otimo[0]), 0.2823529411764706, places=12)

    def test_segundo_grau_2d_acoplada(self):
        """Teste 3: Função quadrática de 2º grau 2x2 não-diagonal (converge em 2 passos)."""
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([-1.0, -2.0])
        x0 = np.array([10.0, 10.0])

        x_otimo, historico = gradiente_conjugado(A, b, x0, verbose=False)
        esperado = -np.linalg.solve(A, b)

        np.testing.assert_allclose(x_otimo, esperado, rtol=1e-7, atol=1e-8)
        self.assertLessEqual(len(historico) - 1, 2)

    def test_segundo_grau_3d_matriz_3x3(self):
        """Teste 4: Função quadrática em 3 variáveis (3x3 simétrica definida positiva)."""
        A = np.array([
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ])
        b = np.array([-5.0, -5.0, -3.0])
        x0 = np.array([10.0, -5.0, 8.0])

        x_otimo, historico = gradiente_conjugado(A, b, x0, verbose=False)
        esperado = np.array([1.0, 1.0, 1.0])

        np.testing.assert_allclose(x_otimo, esperado, rtol=1e-7, atol=1e-8)
        self.assertLessEqual(len(historico) - 1, 3)

    def test_terceiro_grau_tabela_exata_relatorio(self):
        """
        Teste 5 (Tabela 3.4.2 do Relatório - Funções Exatas):
        f(x) = -x^3 + 4x^2 + 3x + 4, partindo de x0 = 0 -> x* = -1/3 ≈ -0.3333333.
        """
        f = lambda x: float(-x[0]**3 + 4.0 * x[0]**2 + 3.0 * x[0] + 4.0)
        grad_f = lambda x: np.array([-3.0 * x[0]**2 + 8.0 * x[0] + 3.0])

        for beta in ("FR", "PR"):
            x_otimo, _ = gradiente_conjugado_nao_linear(
                f, grad_f, np.array([0.0]), metodo_beta=beta, metodo_busca="secao_aurea", verbose=False
            )
            self.assertAlmostEqual(float(x_otimo[0]), -1.0 / 3.0, places=5)

    def test_quarto_grau_tabela_exata_relatorio(self):
        """
        Teste 6 (Tabela 3.4.2 do Relatório - Funções Exatas):
        f(x) = x^4 - 4x^3 + 6x^2 - 4x + 1 = (x - 1)^4, partindo de x0 = 0 -> x* = 1.0.
        """
        f = lambda x: float((x[0] - 1.0) ** 4)
        grad_f = lambda x: np.array([4.0 * (x[0] - 1.0) ** 3])

        x_otimo, _ = gradiente_conjugado_nao_linear(
            f, grad_f, np.array([0.0]), metodo_beta="FR", metodo_busca="secao_aurea", verbose=False
        )
        self.assertAlmostEqual(float(x_otimo[0]), 1.0, places=4)

    def test_terceiro_grau_tabela_inexata_relatorio(self):
        """
        Teste 7 (Tabela 3.4.2 do Relatório - Funções Inexatas com Armijo):
        f(x) = -x^3 + 6x^2 + 20x + 3, partindo de x0 = 0 -> x* = (6 - sqrt(96))/3 ≈ -1.265986.
        """
        f = lambda x: float(-x[0]**3 + 6.0 * x[0]**2 + 20.0 * x[0] + 3.0)
        grad_f = lambda x: np.array([-3.0 * x[0]**2 + 12.0 * x[0] + 20.0])

        x_otimo, _ = gradiente_conjugado_nao_linear(
            f, grad_f, np.array([0.0]), metodo_beta="PR", metodo_busca="armijo", verbose=False
        )
        esperado = (6.0 - np.sqrt(96.0)) / 3.0  # ≈ -1.2659863237
        self.assertAlmostEqual(float(x_otimo[0]), esperado, places=5)

    def test_terceiro_grau_2d_newton_cg(self):
        """
        Teste 8: Polinômio de 3º grau em 2 variáveis:
        f(x1, x2) = (1/3)*(x1^3 + x2^3) + 2*x1^2 + 1.5*x2^2 - 5*x1 - 4*x2 -> x* = [1.0, 1.0].
        """
        grad_cubico = lambda x: np.array([x[0]**2 + 4.0 * x[0] - 5.0, x[1]**2 + 3.0 * x[1] - 4.0])
        hess_cubico = lambda x: np.array([[2.0 * x[0] + 4.0, 0.0], [0.0, 2.0 * x[1] + 3.0]])

        x_otimo, _ = gradiente_conjugado_terceiro_grau(
            grad_cubico, hess_cubico, np.array([5.0, 5.0]), verbose=False
        )
        np.testing.assert_allclose(x_otimo, np.array([1.0, 1.0]), rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)

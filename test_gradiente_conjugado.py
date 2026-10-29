"""
Testes Automatizados — Método dos Gradientes Conjugados (2º e 3º Grau)
======================================================================

Para executar todos os testes no terminal:
    python -m unittest test_gradiente_conjugado.py -v
"""

import unittest
import numpy as np
from gradiente_conjugado_limpo import (
    gradiente_conjugado,
    gradiente_conjugado_terceiro_grau,
)


class TestGradienteConjugado(unittest.TestCase):
    """Bateria de testes para funções de 2º grau (2D e 3D) e de 3º grau (cúbicas)."""

    def test_segundo_grau_diagonal_tcc(self):
        """Teste 1: Função quadrática de 2º grau diagonal (dados originais do TCC)."""
        A = np.array([[15.0, 0.0], [0.0, 15.0]])
        b = np.array([8.0, 3.0])
        x0 = np.array([10.0, 10.0])

        x_otimo, historico = gradiente_conjugado(A, b, x0, verbose=False)
        esperado = np.array([-8.0 / 15.0, -3.0 / 15.0])

        np.testing.assert_allclose(x_otimo, esperado, rtol=1e-7, atol=1e-8)
        # Matriz múltipla da identidade deve convergir em apenas 1 passo (histórico = [x0, x1])
        self.assertEqual(len(historico) - 1, 1)

    def test_segundo_grau_2d_acoplada(self):
        """Teste 2: Função quadrática de 2º grau 2x2 não-diagonal (converge em 2 passos)."""
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([-1.0, -2.0])
        x0 = np.array([10.0, 10.0])

        x_otimo, historico = gradiente_conjugado(A, b, x0, verbose=False)
        esperado = -np.linalg.solve(A, b)  # [1/11, 7/11]

        np.testing.assert_allclose(x_otimo, esperado, rtol=1e-7, atol=1e-8)
        self.assertLessEqual(len(historico) - 1, 2)

    def test_segundo_grau_3d_matriz_3x3(self):
        """Teste 3: Função quadrática em 3 variáveis (3x3 simétrica definida positiva)."""
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
        # Em dimensão n = 3, converge em no máximo 3 passos
        self.assertLessEqual(len(historico) - 1, 3)

    def test_terceiro_grau_polinomio_cubico_desacoplado(self):
        """
        Teste 4: Polinômio de 3º grau (cúbico):
        f(x1, x2) = (1/3)*(x1^3 + x2^3) + 2*x1^2 + 1.5*x2^2 - 5*x1 - 4*x2
        Mínimo local na região convexa (x > 0): x* = [1.0, 1.0].
        """
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

        x0 = np.array([5.0, 5.0])
        x_otimo, _ = gradiente_conjugado_terceiro_grau(grad_cubico, hess_cubico, x0, verbose=False)
        esperado = np.array([1.0, 1.0])

        np.testing.assert_allclose(x_otimo, esperado, rtol=1e-6, atol=1e-7)

    def test_terceiro_grau_polinomio_cubico_acoplado(self):
        """
        Teste 5: Polinômio de 3º grau com variáveis acopladas:
        f(x1, x2) = (1/3)*(x1 - 2)^3 + (1/3)*(x2 - 3)^3 + 2*(x1 - 2)^2 + (x1 - 2)*(x2 - 3) + 3*(x2 - 3)^2
        Para x1 >= 2, x2 >= 3, o único ponto estacionário com gradiente zero é x* = [2.0, 3.0].
        """
        def grad_acoplado(x: np.ndarray) -> np.ndarray:
            u = x[0] - 2.0
            v = x[1] - 3.0
            return np.array([
                u**2 + 4.0 * u + v,
                v**2 + u + 6.0 * v,
            ])

        def hess_acoplado(x: np.ndarray) -> np.ndarray:
            u = x[0] - 2.0
            v = x[1] - 3.0
            return np.array([
                [2.0 * u + 4.0, 1.0],
                [1.0, 2.0 * v + 6.0],
            ])

        x0 = np.array([6.0, 8.0])
        x_otimo, _ = gradiente_conjugado_terceiro_grau(grad_acoplado, hess_acoplado, x0, verbose=False)
        esperado = np.array([2.0, 3.0])

        np.testing.assert_allclose(x_otimo, esperado, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)

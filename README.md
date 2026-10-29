# 📐 Um Estudo Sobre o Método de Direções Conjugadas para Otimização Irrestrita

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jpM90kuKxQ6HxFstiut75V1-EG-yoNHj)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-C%C3%A1lculo%20Num%C3%A9rico-013243?logo=numpy&logoColor=white)
![UFAM](https://img.shields.io/badge/UFAM-PIBIC%20%2F%20FAPEAM-006633)

Implementação em Python do **Método das Direções Conjugadas**, **Busca Exata pela Seção Áurea** e **Busca Inexata de Armijo** para minimização irrestrita de funções quadráticas (2º grau) e não-quadráticas (3º e 4º grau).

> 💡 **Sobre este projeto:** Este repositório reúne o **meu primeiro código escrito em Python** (desenvolvido em 2021 durante a Iniciação Científica / TCC na **Universidade Federal do Amazonas — UFAM**, projeto `PIB-E/0074/2020` com apoio da **FAPEAM**, sob orientação do **Prof. Dr. Roberto Cristóvão Mesquita Silva**), o [Relatório Final em PDF (`Relatorio_Final_PIBIC_UFAM_2021.pdf`)](./Relatorio_Final_PIBIC_UFAM_2021.pdf) e uma **versão refatorada, limpa e testada** de todos os algoritmos estudados.

---

## 🎯 Fundamentação Matemática

### 1. Caso Quadrático (2º Grau — Seção 3.3.2)
Minimização de uma função quadrática estritamente convexa:

$$f(x) = \frac{1}{2} x^T A x + b^T x + c$$

onde $A \in \mathbb{R}^{n \times n}$ é simétrica definida positiva ($A > 0$). O método gera direções $A$-conjugadas ($(d^i)^T A d^j = 0, \forall i \neq j$) que garantem encontrar o minimizador global $x^*$ em **no máximo $n$ iterações**:

* **Tamanho exato do passo ($t_k$):**
  $$t_k = -\frac{\nabla f(x^k)^T d^k}{(d^k)^T A d^k}$$
* **Coeficiente de conjugação ($\beta_k$):**
  $$\beta_k = \frac{(d^k)^T A \nabla f(x^{k+1})}{(d^k)^T A d^k}$$

### 2. Extensão Não-Quadrática — 3º e 4º Grau (Seção 3.3.3)
Para funções gerais $f: \mathbb{R}^n \to \mathbb{R}$ de 3º grau (cúbicas), 4º grau ou superiores:
1. **Busca Unidirecional para o passo $t_k$ (Seção 3.2):**
   * **Busca Exata (Método da Seção Áurea em 2 Fases):** Encontra o intervalo $[a,b]$ e reduz pela razão áurea ($\theta_1 = \frac{3-\sqrt{5}}{2} \approx 0,382$ e $\theta_2 = \frac{\sqrt{5}-1}{2} \approx 0,618$).
   * **Busca Inexata (Condição de Armijo):** Encontra $t = \gamma^m$ tal que $f(x^k + t d^k) \le f(x^k) + \eta t \nabla f(x^k)^T d^k$.
2. **Fórmulas para $\beta_k$ com reinício a cada $n$ passos ($(k+1) \bmod n = 0$):**
   * **Fletcher-Reeves (1964):**
     $$\beta_k^{FR} = \frac{\|\nabla f(x^{k+1})\|^2}{\|\nabla f(x^k)\|^2}$$
   * **Polak-Ribière (1969):**
     $$\beta_k^{PR} = \frac{\nabla f(x^{k+1})^T (\nabla f(x^{k+1}) - \nabla f(x^k))}{\|\nabla f(x^k)\|^2}$$

---

## 🔄 Evolução do Código (Original de 2021 vs. Versão Refatorada)

| Aspecto | Código Original (`Algoritomo_...`) | Versão Limpa (`gradiente_conjugado_limpo.py`) |
| :--- | :--- | :--- |
| **Multiplicação Matricial** | `np.dot(np.dot(d.T, A), d)` | Operador nativo `@`: `d @ A @ d` |
| **Critério de Parada** | Função `comparar(g)` com `for` e strings `"igual"` / `"Não igual"` | Norma euclidiana do gradiente: `np.linalg.norm(g) < tol` |
| **Busca Linear ($t_k$)** | Apenas fórmula quadrática com $A$ fixo | **Seção Áurea (2 fases)**, **Busca de Armijo** e **Fórmula Exata** |
| **Variantes de $\beta_k$** | Apenas fórmula matricial com $A$ | **Matricial ($A$)**, **Fletcher-Reeves (`FR`)** e **Polak-Ribière (`PR`)** |

---

## 📂 Estrutura do Repositório

```text
├── Relatorio_Final_PIBIC_UFAM_2021.pdf           # Monografia / Relatório Final completo (UFAM / FAPEAM)
├── Gradiente_Conjugado_Limpo.ipynb               # Notebook didático com gráficos de convergência (2º e 3º grau)
├── gradiente_conjugado_limpo.py                  # Módulo completo (Quadrático, Seção Áurea, Armijo, FR, PR, Newton-CG)
├── test_gradiente_conjugado.py                   # Suíte de 8 testes automatizados (incluindo a Seção 3.4 do Relatório)
├── Algoritomo_conjugados_quadraticos_teste.ipynb # Notebook original de 2021 (compatível com Colab)
├── Algoritomo_conjugados_quadraticos_teste.py    # Primeiro script Python original (2021)
└── README.md                                     # Documentação do projeto
```

---

## 🧪 Casos de Teste da Seção 3.4 do Relatório (`test_gradiente_conjugado.py`)

A suíte de testes valida automaticamente tanto funções quadráticas quanto os polinômios de 3º e 4º grau da **Seção 3.4 (Teste Computacional)** do relatório:

| Teste | Grau / Dimensão | Função / Parâmetros | Ponto Inicial ($x^0$) | Mínimo Encontrado ($x^*$) |
| :--- | :---: | :--- | :---: | :---: |
| **1. TCC Diagonal** | 2º Grau ($\mathbb{R}^2$) | $A = 15I, \; b = [8, 3]^T$ | $[10, 10]^T$ | $[-0.533333, -0.2]^T$ (1 passo) |
| **2. Relatório 3.4.1** | 2º Grau ($\mathbb{R}^1$) | $A = [42.5], \; b = [-12]$ | $10$ | $0.2823529411764706$ (1 passo) |
| **3. Elíptica 2D** | 2º Grau ($\mathbb{R}^2$) | $A = [[4, 1], [1, 3]], \; b = [-1, -2]^T$ | $[10, 10]^T$ | $[0.090909, 0.636364]^T$ (2 passos) |
| **4. Tridimensional** | 2º Grau ($\mathbb{R}^3$) | Matriz $3 \times 3$ simétrica definida positiva | $[10, -5, 8]^T$ | $[1, 1, 1]^T$ (3 passos) |
| **5. Relatório 3.4.2 (Cúbica A)** | 3º Grau ($\mathbb{R}^1$) | $f(x) = -x^3 + 4x^2 + 3x + 4$ (Seção Áurea, FR/PR) | $0$ | $-0.3333333$ ($-1/3$) |
| **6. Relatório 3.4.2 (Quártica)** | 4º Grau ($\mathbb{R}^1$) | $f(x) = x^4 - 4x^3 + 6x^2 - 4x + 1 = (x-1)^4$ | $0$ | $1.000000$ |
| **7. Relatório 3.4.2 (Cúbica B)** | 3º Grau ($\mathbb{R}^1$) | $f(x) = -x^3 + 6x^2 + 20x + 3$ (Armijo, PR) | $0$ | $-1.265986$ |
| **8. Cúbica 2D** | 3º Grau ($\mathbb{R}^2$) | $f(x_1, x_2) = \frac{1}{3}(x_1^3+x_2^3) + 2x_1^2 + 1.5x_2^2 - 5x_1 - 4x_2$ | $[5, 5]^T$ | $[1.0, 1.0]^T$ |

---

## 🚀 Como Executar

```bash
# Executar os exemplos do Relatório PIBIC
python gradiente_conjugado_limpo.py

# Rodar a bateria de 8 testes automatizados
python -m unittest test_gradiente_conjugado.py -v
```

---

## 👤 Autor e Créditos Acadêmicos

* **Autor:** Francisco Lima da Silva (`Chico-Lima-Silva`)
* **Orientador:** Prof. Dr. Roberto Cristóvão Mesquita Silva
* **Instituição:** Universidade Federal do Amazonas (UFAM) — Programa Institucional de Iniciação Científica (`PIB-E/0074/2020`)
* **Financiamento:** Fundação de Amparo à Pesquisa do Estado do Amazonas (FAPEAM)

# Algoritmo de Gradientes Conjugados (TCC)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jpM90kuKxQ6HxFstiut75V1-EG-yoNHj)

Este repositório contém a implementação em Python do **Método dos Gradientes Conjugados** aplicado à minimização de funções quadráticas.

> 💡 **Nota histórica:** Este foi o meu primeiro código escrito em Python, desenvolvido como parte do meu **Trabalho de Conclusão de Curso (TCC)**.

---

## 📐 Descrição Matemática

O algoritmo busca encontrar o ponto de mínimo $x^*$ de uma função quadrática convexa da forma:

$$f(x) = \frac{1}{2} x^T A x + b^T x + c$$

onde $A \in \mathbb{R}^{n \times n}$ é uma matriz simétrica definida positiva, $b \in \mathbb{R}^n$ e $c \in \mathbb{R}$.

A cada iteração $k$:
1. **Gradiente:** $g_k = \nabla f(x_k) = A x_k + b$
2. **Tamanho do passo ($t_k$):**
   $$t_k = -\frac{g_k^T d_k}{d_k^T A d_k}$$
3. **Atualização do ponto ($x_{k+1}$):**
   $$x_{k+1} = x_k + t_k d_k$$
4. **Coeficiente de conjugação ($\beta_k$):**
   $$\beta_k = \frac{d_k^T A g_{k+1}}{d_k^T A d_k}$$
5. **Nova direção conjugada ($d_{k+1}$):**
   $$d_{k+1} = -g_{k+1} + \beta_k d_k$$

---

## 📂 Estrutura do Repositório

* `Algoritomo_conjugados_quadraticos_teste.ipynb` — Notebook interativo (Jupyter / Google Colab) com o passo a passo do algoritmo e versão atualizada para o NumPy moderno.
* `Algoritomo_conjugados_quadraticos_teste.py` — Script Python original exportado na época do TCC.

---

## 🚀 Como Executar

### Opção 1: Google Colab (no navegador)
Clique no botão **Open in Colab** no topo deste README para abrir e executar o notebook diretamente na nuvem.

### Opção 2: Localmente
1. Instale o NumPy:
   ```bash
   pip install numpy jupyter
   ```
2. Abra o notebook:
   ```bash
   jupyter notebook Algoritomo_conjugados_quadraticos_teste.ipynb
   ```

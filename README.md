# Monte Carlo Simulation – Euler’s Constant

## Contexte

L’objectif de ce projet est d’estimer la constante d’Euler :

$$
\gamma = \lim_{n\to\infty}\left(\sum_{k=1}^{n}\frac{1}{k}-\log n\right).
$$

Elle admet aussi les représentations suivantes, utiles pour construire des estimateurs :

$$
\gamma = \sum_{k=1}^{\infty}\left(\frac{1}{k}-\log\!\left(1+\frac{1}{k}\right)\right), 
\qquad
\gamma = \int_{0}^{1} -\log(-\log x)\,dx.
$$

Ces formulations permettent d’explorer plusieurs approches de Monte Carlo pour estimer $\gamma$ et comparer biais, variance et efficacité.

## Questions du sujet

1. **Importance sampling**  
   Construire un estimateur sans biais de $\gamma$ à partir de la représentation en série.

2. **Monte Carlo standard (MC)**, **Stratified MC** et **Quasi–Monte Carlo (QMC)**   
   En utilisant la représentation sous forme d'intégrale de la constante, construire un estimateur avec chacune de ces méthodes et visualiser les différentes vitesses de convergence.

3. **Control variates**  
   Introduire des variables de contrôle pour réduire la variance des estimateurs de la question 2.

4. **Estimateur à somme tronquée**  
   Utiliser un estimateur du type $\sum_{k=1}^{R} a_k/\mathbb P(R\ge k)$ pour estimer sans biais la constante sous forme de série.

## Résultats

### Ordres de convergence (régression log–log des erreurs)

| Méthode                              | Ordre approx. | $R^2$ |
|--------------------------------------|---------------|-------|
| Truncated Sum                         | -0.221        | 0.927 |
| Importance Sampling                   | -0.502        | 0.996 |
| Standard Monte Carlo                  | -0.503        | 0.997 |
| MC + Control Variates                 | -0.628        | 0.968 |
| Quasi Monte Carlo                     | -0.860        | 0.998 |
| Stratified Monte Carlo                | -1.036        | 0.999 |
| QMC + Control Variates                | -1.068        | 0.988 |

## Conclusion

Les méthodes les plus performantes observées sont **QMC avec control variates** et **Stratified MC**, qui offrent le meilleur compromis coût/précision pour l’estimation de $\gamma$.

## Installation
`numpy`, `scipy`, `matplotlib`

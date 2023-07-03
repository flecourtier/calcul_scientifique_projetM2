---
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
---
\begin{algorithm}[H]
\DontPrintSemicolon
\SetAlgoLined
\SetKwInOut{Input}{Input}\SetKwInOut{Output}{Output}
\Input{a tolerance $\epsilon_{tol}$ and $S_1=\mu_1$  }
\Output{a sample set $S_N$}
\BlankLine
$X_1=Span \{u(\mu_1)\}$\;
\While{$\Delta_{N}^{max}\ge \epsilon^{tol}$}{
    N=N+1\;
    $\mu_N=\arg\max\frac{\Delta_{N-1}^{en}(\mu)}{w(\mu)}$\;
    $\Delta_N^{max}=\max\frac{\Delta_{N-1}^{en}(\mu)}{w(\mu)}$\;
    $S_N=S_{N-1}\cup\{\mu_N\}$\;
    $X_N=X_{N-1}+Span \{u(\mu_N)\}$\;
}
\caption{Greedy Sampling Procedure}
\end{algorithm} 

---
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
---
\begin{algorithm}[H]
\DontPrintSemicolon
\SetAlgoLined
\SetKwInOut{Input}{Input}\SetKwInOut{Output}{Output}
\Input{a tolerance $\epsilon_{tol}$ and $\mu_0^*$}
\Output{a sample set $S_N$ and $X_N$}
\BlankLine
$X_N=\{0\}, S_N=\{0\},N=0, \mu^*=\mu_0^*$\;
\While{$\Delta_{N}^{max}\ge \epsilon^{tol}$}{
    $e_{N,proj}^k(\mu^*)=u^k(\mu^*)-proj_{X,X_N}u^k(\mu^*), 1\le k\le K$\;
    $S_{N+1}=S_{N}\cup\{\mu^*\}$\;
    $X_{N+1}=X_N+POD_X(\{e_{N,proj}^k(\mu^*), 1\le k\le K\},1)$\;
    $N=N+1$\;
    $\mu^*=\arg\max_{\mu\in\Xi_{train}}\frac{\Delta_{N}^{K}(\mu)}{|||u_N^K(\mu)|||}$\;
    $\Delta_N^{max}=\frac{\Delta_{N}^{K}(\mu^*)}{|||u_N^K(\mu^*)|||}$\;
}
\caption{POD-Greedy Sampling Procedure}
\end{algorithm} 
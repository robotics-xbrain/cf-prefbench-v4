# 5 A Sanity Diagnostic Battery for Compositional Preference Methods

Section~4 reports a phenomenon — the lexical cliff — that we discovered *because* a six-probe diagnostic battery flagged an earlier "positive" result of ours as a centroid pass-through. The same battery later flagged an "anti-correlation" interpretation of our Exp~1 PFA $=0.000$ as an aggregation artifact (Section~4.2). We present the battery here as a reusable contribution: six lightweight probes that operate on a trained preference model and detect three failure modes — pass-through pathways, gate triviality, and instruction non-use — that produce indistinguishable accuracy curves under standard evaluation.

## 5.1 Motivation: A False Positive We Caught Ourselves

Before the work in Section~4, we trained a hybrid LIB+centroid model that scored $0.905 \pm 0.034$ on the v3 \texttt{test\_heldout\_color} color-axis subset, just shy of the engineered ceiling. The model used a learned gate to blend a CLIP-attention path (LIB) with a 173-dim engineered centroid feature. We initially reported this as a positive result. Six lightweight probes — described below — revealed that the LIB path contributed nothing: zeroing the centroid feature dropped accuracy to $0.167$, while zeroing the LIB path left accuracy at $0.929$. The hybrid was a centroid pass-through with a decorative cross-attention module that the head had learned to ignore.

This is a representative failure mode for compositional preference methods with multiple signal pathways. A learned blend over a strong engineered feature and a weaker learned feature will, under unconstrained training, collapse to the strong path while the weaker path stays present in the forward computation but contributes nothing to the gradient that updates the head. Without targeted probes, the result is indistinguishable from a true hybrid synergy on standard test accuracy. We made this mistake on our own architecture and corrected it with the probes in Section~5.2; we present the probes here so others can avoid the same false positive.

## 5.2 The Six Probes

For a preference model with a CLIP visual path, optional engineered centroid path, optional gate, and instruction text input, the battery is:

\begin{table}[h]
\centering
\small
\begin{tabular}{lll}
\toprule
Probe & Perturbation & What it tests \\
\midrule
SC-1 & Force gate $=1$ (LIB-only)            & Pure-learned-path accuracy \\
SC-2 & Force gate $=0$ (centroid-only)       & Pure-engineered-path accuracy \\
SC-3 & Random gate per call                  & Gate's functional relevance \\
SC-4 & Zero engineered centroid feature      & Reliance on engineered signal \\
SC-5 & Zero learned (LIB) feature            & Reliance on learned signal \\
SC-6 & Shuffle CLIP text within batch        & Instruction usage \\
\bottomrule
\end{tabular}
\caption{The six sanity probes. SC-1..SC-3 are gate-architecture-specific; SC-4..SC-6 apply to any model with separable feature pathways and an instruction input.}
\label{tab:sanity-probes}
\end{table}

An honest LIB+centroid hybrid should show: SC-1 $\approx$ LIB-only baseline, SC-2 $\approx$ centroid-only baseline, SC-3 between the two, SC-4 $\ll$ NORMAL (centroid contributes), SC-5 $\ll$ NORMAL (LIB contributes), and SC-6 $\ll$ NORMAL (instruction used). Any probe that returns NORMAL accuracy indicates that the perturbed signal pathway is not actually carrying information. This is unambiguous when the NORMAL accuracy is high and one probe leaves it unchanged.

## 5.3 Case Study 1: Detecting Centroid Pass-Through

\begin{table}[h]
\centering
\small
\begin{tabular}{lrll}
\toprule
Probe & PFA & Expected (honest hybrid) & Verdict \\
\midrule
NORMAL & 0.929 & — & matches paper claim \\
SC-1 force gate=1 & 0.143 & $\sim$0.57–0.64 (LIB-only) & \textbf{FAIL} (well below) \\
SC-2 force gate=0 & 0.929 & $\sim$0.929 (centroid-only) & PASS \\
SC-3 random gate & 0.923 & in $[0.64, 0.93]$ & at top of range \\
SC-4 zero centroid & 0.167 & $\ll$ NORMAL if centroid contributes & \textbf{centroid carries 100\%} \\
SC-5 zero LIB & 0.929 & $\ll$ NORMAL if LIB contributes & \textbf{LIB unused} \\
SC-6 shuffle text & 0.929 & $\ll$ NORMAL if text used & CLIP text path unused$^\dagger$ \\
\bottomrule
\end{tabular}
\caption{Phase 3 hybrid v3 \texttt{test\_heldout\_color} color-axis PFA, $n=14$ flip groups, seed 42, ensemble eval. $^\dagger$SC-6 shuffles the CLIP text embedding only; the centroid pair feature already encodes the instruction-conditioned target color via the engineered parsing pipeline, so SC-6 understates instruction usage in this hybrid. The other probes are unambiguous.}
\label{tab:phase3-sanity}
\end{table}

The probes converge on a single diagnosis: the centroid pair feature carries the entire discriminative signal. SC-4 (zero centroid) collapses accuracy from $0.929$ to $0.167$. SC-5 (zero LIB patches) leaves accuracy at $0.929$. SC-1 (LIB-only at gate $=1$) drops to $0.143$, below the LIB-only baseline of $0.643$ reported in Section~3, because the head was trained against gate values near $0.5$ and never saw pure LIB-path predictions. The gate is functionally irrelevant: SC-3 (random gate) reaches $0.923$, same as NORMAL. This is the centroid pass-through diagnosed at the architectural level: with `binding_a` and `binding_b` available to the head, plus a centroid-derived `final_diff` that the head can amplify, LIB's binding scores are simply unused.

## 5.4 Case Study 2: Validating LIB+L/14 Closes the Cliff

For pure-LIB architectures without an engineered centroid (Section~4.3), SC-1, SC-2, SC-3, and SC-4 are not informative — there is no gate and no centroid. The relevant probes are SC-5 (zero LIB patches) and a stronger lexical variant: \emph{substitution}, where the held-out instruction's CLIP text feature is replaced with a training-verb instruction with identical binding tokens. The substitution probe is a stricter version of SC-6 that holds the binding signal constant while perturbing only the lexical surface form (Appendix~A).

\begin{table}[h]
\centering
\small
\begin{tabular}{lrr}
\toprule
Probe & motion (B/32) & motion (L/14) \\
\midrule
NORMAL row acc & 0.750$^\ddagger$ & 0.917 \\
SC-6 ZERO text feature & 0.500 & 0.500 \\
SC-6 SHUFFLE text feature & 0.452 & — \\
SUBSTITUTION (train-verb text) & 0.893 & 0.917 \\
\bottomrule
\end{tabular}
\caption{Pure-LIB sanity probes on the motion\_sequence axis. NORMAL is the mean over \{shift, convey, transit\}; substitution restores transit (the cliff verb) on B/32 from $0.536$ to $0.893$, isolating the failure to text-encoding. Under L/14 the held-out verbs already pass; substitution is therefore uninformative (it cannot lift an already-correct prediction). $^\ddagger$0.750 is the seed-1 retrain used for the sanity battery; the 3-seed B/32 mean reported in Section~4.2 is $0.786$.}
\label{tab:l14-sanity}
\end{table}

SC-6 (zero / shuffle text) collapses both B/32 and L/14 to $\leq 0.50$, confirming both models genuinely use the instruction. Substitution recovers B/32 from $0.750$ to $0.893$ but is uninformative on L/14 because there is no failure to recover from. The combination — substitution lifts B/32 dramatically but cannot lift L/14, and SC-6 collapses both — supports the Section~4.3 claim that L/14's text-embedding capacity, not its visual-attention, is the relevant difference.

## 5.5 Case Study 3: Falsifying Four Anti-Collapse Architectures

After the Phase 3 pass-through diagnosis, we tested four architectural mechanisms designed to force LIB to contribute: information bottleneck on the centroid (B1), frozen-residual structural separation (B2), KL output-divergence penalty (B3), and adversarial discrimination of pathway outputs (B4 at two regularization strengths). The success criteria, set before training: NORMAL $\geq 0.75$, SC-5 drop $\geq 0.15$ (zeroing LIB must hurt), SC-6 drop $\geq 0.10$ (shuffling text must hurt).

\begin{table}[h]
\centering
\small
\begin{tabular}{lrrr}
\toprule
Variant & NORMAL & SC-5 drop & SC-6 drop \\
\midrule
B1 bottleneck & 0.286 & $-0.571$ (LIB \emph{hurts}) & $+0.071$ \\
B2 frozen residual & 0.357 & n/a (residual is detached) & $+0.357$ \\
B3 MI penalty & 0.857 & $-0.071$ & $+0.087$ \\
B4 adv. $\lambda{=}0.3$ & 0.786 & $+0.071$ & $+0.286$ \\
B4 adv. $\lambda{=}0.5$ & 0.857 & $0.000$ & $0.000$ \\
\bottomrule
\end{tabular}
\caption{Phase 4 anti-collapse sanity matrix. Bold targets: NORMAL $\geq 0.75$, SC-5 drop $\geq 0.15$, SC-6 drop $\geq 0.10$. No variant passes all three. B4 $\lambda{=}0.3$ comes closest (passes NORMAL and SC-6 but only $+0.07$ on SC-5).}
\label{tab:anti-collapse}
\end{table}

The sanity battery converts a noisy ranking on NORMAL accuracy into an unambiguous falsification: even the highest-NORMAL variants (B3, B4-$\lambda{=}0.5$, both at $0.857$) fail SC-5 entirely (zeroing LIB does not hurt). Three of the four mechanisms are either inactive (B3, B4-$\lambda{=}0.5$) or actively harmful (B1's negative SC-5 drop indicates the bottlenecked centroid forced the head to use LIB for parts of the signal in a way that hurt aggregate accuracy). The mechanistic conclusion — \emph{differentiable hybrid architectures collapse to the engineered signal pathway, robust to four anti-collapse interventions} — would be invisible if we reported only NORMAL accuracy.

## 5.6 Architectural Caveats

Not every probe is informative for every architecture. SC-1..SC-3 require a gate between learnable pathways and are vacuous for single-pathway models. SC-4 requires a separable engineered feature; it is trivially destructive when applied to a model whose only input is the centroid. SC-5 is trivially destructive for a pure-CLIP-feature model with no other pathway. SC-6 (zero/shuffle text) is informative whenever the model accepts an instruction input. For pure-LIB models without engineered centroids, the informative battery reduces to SC-5 (zero patch features, expected to be trivially destructive — sanity-checks the model is using the visual input) and SC-6 plus the stronger substitution probe that holds binding tokens constant while changing lexical surface form. Practitioners should select the probes whose perturbations target their architecture's actual signal pathways.

## 5.7 Recommendations

Run SC-4..SC-6 on any compositional preference model before claiming a positive result. For models with multiple feature pathways, also run SC-1..SC-3 to detect gate triviality. Report the full sanity matrix alongside headline accuracy, even when all probes pass. We hope this becomes routine reporting for compositional preference work; in our case, the battery transformed two near-positive results (Phase 3 hybrid, four Phase 4 anti-collapse variants) and one near-anti-correlation result (Section~4.2 PFA $=0.000$) into honest negatives, while validating that L/14 cliff closure (Section~4.3) is text-side and stable. The battery cost roughly thirty minutes of compute per model in our setup, dominated by training-time CLIP feature extraction; the probes themselves are fast.

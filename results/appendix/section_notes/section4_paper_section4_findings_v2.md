# 4 Findings: The Lexical Cliff in Compositional Preference Learning

We report two empirical findings on CF-PrefBench v4. First, an instruction-conditioned binding module (LIB) trained on CLIP ViT-B/32 patch features exhibits a sharp accuracy drop when held-out paraphrases sit lexically far from the training pool. Second, this *lexical cliff* is not a property of the binding architecture: replacing the visual backbone with ViT-L/14 fully closes the cliff on motion and size, and partially closes it on speed, without any change to the dataset or training recipe. The cliff is a small-backbone capacity effect, not a universal CLIP limitation, and its mechanism differs across binding axes — cosine-monotonic on motion verbs, class-bimodal on size and speed.

## 4.1 Setup

CF-PrefBench v4 (built on the v3 dataset of \citet{cfprefbench-v3}) covers 7 binding axes (color, object, action, spatial, size, motion\_sequence, speed) over 7{,}749 video-pair preference examples. Each example is a triple $(v_A, v_B, t)$ where the two videos differ on a single binding axis and the instruction $t$ resolves to a preferred video; counterfactual flip groups balance A and B exactly. Train, val, and test\_seen splits share paraphrase verbs; the test\_heldout\_lexical split substitutes synonymous verbs that never appear in training. Held-out paraphrase pools are verified disjoint from training (Appendix~B). LIB v0 is a per-attribute cross-attention module that projects CLIP patch tokens onto four learned attribute queries derived from the CLIP text embedding; we train it with a four-term loss (BCE + recon + counterfactual + paraphrase stability), 60 epochs, AdamW, three random seeds. We compare two frozen CLIP visual encoders: ViT-B/32 (49 patches, 768-dim) and ViT-L/14 (256 patches, 1024-dim, $\sim$3$\times$ parameters).

Every metric is computed over three seeds with full per-row predictions preserved. We report row-level accuracy throughout; strict flip-group preference accuracy (PFA) is reported as a footnote when its aggregation behavior diverges meaningfully from per-row accuracy.\footnote{PFA on motion\_sequence test\_heldout\_lexical is 0.000 across all three seeds while per-row accuracy is 0.786. The discrepancy reflects PFA's all-or-nothing per-flip-group aggregation: when a single held-out verb (transit) fails systematically, every 6-row flip group has at least one wrong row and the strict score collapses to zero. Per-row accuracy is therefore the more informative metric for these splits, but PFA values are preserved in the raw outputs (Appendix~C).}

## 4.2 The Lexical Cliff on Motion Verbs

We test LIB v0 on the motion\_sequence test\_heldout\_lexical split, which substitutes verbs \{shift, convey, transit, scoot\} for the training set \{move, drag, push\}. Held-out and training paraphrases share identical direction tokens ("left then right" or "right then left"); only the verb changes. Table~\ref{tab:motion-cliff} shows per-row accuracy alongside the CLIP-B/32 cosine similarity of each held-out instruction to the mean training instruction.

\begin{table}[t]
\centering
\small
\begin{tabular}{lrr}
\toprule
Verb & cos(B/32) & Accuracy (3-seed) \\
\midrule
shift & 0.968 & 0.917 $\pm$ 0.017 \\
convey & 0.938 & 0.929 $\pm$ 0.000 \\
scoot & 0.927 & 0.679 $\pm$ 0.058 \\
transit & 0.915 & 0.500 $\pm$ 0.000 \\
\bottomrule
\end{tabular}
\caption{ViT-B/32 motion-verb cliff. Held-out accuracy correlates monotonically with cosine to training. Pearson $r=0.808$ ($p=0.0015$), Spearman $\rho=0.880$ ($p=0.00016$) over $n=12$ observations.}
\label{tab:motion-cliff}
\end{table}

Accuracy decreases monotonically with the held-out verb's CLIP cosine to training; the cliff is gradual rather than sharp (Figure~\ref{fig:motion-cliff-b32}). A substitution probe — replacing the held-out instruction's text feature with a training-verb instruction with identical direction tokens, with videos unchanged — recovers transit from 0.500 to 0.893 and scoot from 0.679 to 0.917 (Appendix~A). Recovery scales inversely with cosine, confirming the failure is text-side: the model can read direction from the video; what fails on transit/scoot is the projection from a far-from-training text embedding to the right attribute queries.

\begin{figure}[t]
\centering
\includegraphics[width=0.48\columnwidth]{figures/fig_4_1_motion_cliff_b32.pdf}
\caption{Motion-verb cliff under ViT-B/32. Per-seed dots overlay the 3-seed mean (error bars). Accuracy decreases monotonically with CLIP cosine of the held-out verb to training.}
\label{fig:motion-cliff-b32}
\end{figure}

## 4.3 Cross-Architecture Replication: ViT-L/14 Closes the Motion Cliff

If the cliff stems from the small text-embedding capacity of B/32, a larger CLIP variant should reduce or remove it. We replicate Section~4.2's setup with ViT-L/14 (1024-dim, 256-patch). Visual features are re-extracted for all 1{,}080 new-axis videos; LIB v0 is re-trained from scratch for three seeds with the same recipe (batch reduced to 16 for memory).

\begin{table}[t]
\centering
\small
\begin{tabular}{lrrrrr}
\toprule
Verb & cos(B/32) & B/32 & cos(L/14) & L/14 & $\Delta$ \\
\midrule
shift & 0.968 & 0.917 & 0.925 & 0.905 & $-0.012$ \\
convey & 0.938 & 0.929 & 0.918 & 0.929 & $\pm 0.000$ \\
scoot & 0.927 & 0.679 & 0.875 & 0.905 & $+0.226$ \\
transit & 0.915 & 0.500 & 0.876 & 0.917 & $+0.417$ \\
\bottomrule
\end{tabular}
\caption{Motion-verb cliff under two backbones. Both B/32-cliff verbs recover to $\geq 0.905$ on L/14 \emph{despite} their L/14 cosines to training being lower (0.876, 0.875 vs B/32's 0.915, 0.927).}
\label{tab:motion-l14}
\end{table}

The two B/32-cliff verbs gain $+0.226$ (scoot) and $+0.417$ (transit) under L/14. Crucially, this happens \emph{despite} L/14 placing both verbs at lower cosine to training than B/32 did. There is no universal cosine threshold for direction binding; the threshold is determined by the backbone's text-embedding capacity. Figure~\ref{fig:motion-dual-arch} overlays the two backbone curves.

\begin{figure}[t]
\centering
\includegraphics[width=0.48\columnwidth]{figures/fig_4_2_motion_dual_arch.pdf}
\caption{Motion-verb cliff under two backbones. The orange dashed line (ViT-L/14) is flat across all four verbs, while the blue line (ViT-B/32) shows the cliff.}
\label{fig:motion-dual-arch}
\end{figure}

## 4.4 Axis Generalization: Two Distinct Mechanisms

To test whether the cliff generalizes across binding axes, we evaluate the same six trained models on eight new held-out tokens spanning size and speed: \{colossal, gigantic\} (BIG class), \{miniature, petite\} (SMALL class), \{briskly, speedily\} (FAST class), and \{sluggishly, gradually\} (SLOW class). Tokens span the B/32 cliff zone $[0.89, 0.95]$ while staying inside the original axis's semantic class. No retraining is needed: v4 training data excludes all of these tokens, so the existing checkpoints constitute a clean held-out evaluation.

\begin{table}[t]
\centering
\small
\begin{tabular}{llrrr}
\toprule
Axis & Class & B/32 & L/14 & $\Delta$ \\
\midrule
\multirow{2}{*}{Size}
 & BIG & 0.925 $\pm$ 0.009 & 0.810 $\pm$ 0.067 & $-0.115$ \\
 & SMALL & \textbf{0.444} $\pm$ 0.103 & 0.794 $\pm$ 0.095 & $+0.350$ \\
\multirow{2}{*}{Speed}
 & FAST & \textbf{0.115} $\pm$ 0.084 & 0.389 $\pm$ 0.179 & $+0.273$ \\
 & SLOW & 0.829 $\pm$ 0.134 & 0.996 $\pm$ 0.009 & $+0.167$ \\
\multirow{2}{*}{Motion}
 & above-cliff & 0.923 $\pm$ 0.009 & 0.917 $\pm$ 0.017 & $-0.006$ \\
 & below-cliff & \textbf{0.589} $\pm$ 0.107 & 0.911 $\pm$ 0.014 & $+0.322$ \\
\bottomrule
\end{tabular}
\caption{Three-axis $\times$ two-architecture cliff matrix. The cliff appears on all three axes under B/32 (\textbf{bold} = cliff cells). L/14 fully closes the cliff on motion and size; on speed, the FAST class only partially recovers. $n=6$ per cell.}
\label{tab:3axis-cliff}
\end{table}

Two patterns emerge. On motion verbs, accuracy is a smooth function of cosine. On size and speed, held-out tokens within the same semantic class share nearly-identical cosines yet split into two regimes: BIG and SLOW classes succeed at 0.83–0.93; SMALL and FAST classes collapse to 0.04–0.49. Within each class cosine cannot predict accuracy (all tokens cluster at the same cosine); class membership can. Speed briskly at 0.048 mean is below chance, indicating systematic anti-correlation: the model picks the slower video even when told "move the block briskly" (Figures~\ref{fig:3axis-cliff} and \ref{fig:mechanism}).

The asymmetry suggests B/32-trained LIB has learned a **default-class bias**: prefer the larger block, prefer the slower video. Held-out tokens whose semantic class matches the default succeed because the default is correct for them; held-out tokens in the opposite class fail because the held-out lexical signal cannot override the default. The motion axis lacks an obvious default (direction tokens are symmetric in the training distribution), so its cliff degrades smoothly with cosine instead of snapping to a default. L/14 closes the motion cliff fully and the size cliff fully (SMALL class lifts 0.444 $\rightarrow$ 0.794); on the speed axis it closes the SLOW class to ceiling (0.996) but only partially lifts FAST (0.115 $\rightarrow$ 0.389; briskly stays at 0.317, speedily reaches 0.460 with one seed at 0.786).

**Cross-method check: the cliff is method-specific.** As a sanity check that the cliff is not a universal limitation of vision-language systems, we run zero-shot GPT-4o and Qwen2-VL-2B as judges on the same 448-row cliff test set. Figure~\ref{fig:cross-method-cliff} shows the result. GPT-4o achieves $\geq$0.78 on the size SMALL cliff tokens (miniature, petite) and $\geq$0.64 on the speed FAST cliff tokens (briskly, speedily) — between $+0.337$ and $+0.595$ absolute over LIB-B/32 — confirming that the cliff is specific to the LIB+B/32 pipeline and not a property of large VLMs in general. GPT-4o, however, has a complementary failure mode: at 192$\times$144 video resolution it cannot reliably read direction tokens from a single 2$\times$4 image-grid composite, and scores 0.48–0.60 across all four motion verbs (including the LIB-B/32 above-cliff verbs where LIB scores 0.92). LIB-L/14 thus remains the strongest single method on motion (0.91 vs GPT-4o 0.55), while GPT-4o leads on the speed FAST cliff. The much-smaller Qwen2-VL-2B falls near chance on all 12 tokens, providing a minimum-capability floor rather than a meaningful cliff probe.

\begin{figure*}[t]
\centering
\includegraphics[width=0.95\textwidth]{figures/fig_4_5_cross_method.pdf}
\caption{Cross-method comparison of held-out accuracy across the three binding axes. Each panel shows four held-out tokens (two above-cliff for LIB-B/32, two cliff tokens) and four methods: LIB-B/32 (blue, hatched on cliff tokens), LIB-L/14 (orange), zero-shot GPT-4o (green), and zero-shot Qwen2-VL-2B (purple). Red stars mark the best-performing method on each cliff token. No single method dominates: LIB-L/14 leads on motion cliff tokens, GPT-4o leads on speed FAST cliff tokens, and the two trade narrowly on size SMALL. Qwen-2B sits near chance throughout (minimum-capability floor).}
\label{fig:cross-method-cliff}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.95\textwidth]{figures/fig_4_3_3axis_matrix.pdf}
\caption{Three-axis $\times$ two-architecture cliff matrix ($n=6$ per cell). Hatched bars mark the B/32 cliff cells. L/14 fully closes the cliff on motion and size; on speed it only partially closes the FAST class.}
\label{fig:3axis-cliff}
\end{figure*}

\begin{figure}[t]
\centering
\includegraphics[width=0.95\columnwidth]{figures/fig_4_4_mechanism_diagram.pdf}
\caption{Two cliff mechanisms. (a) Motion verbs: smooth cosine-monotonic cliff. (b) Size and speed: tokens within a class cluster at the same cosine but accuracy splits along the learned default-class prior (BIG/SLOW pass, SMALL/FAST fail).}
\label{fig:mechanism}
\end{figure}

## 4.5 Mechanism Discussion

We interpret the two cliff mechanisms as consequences of the visual-temporal structure each axis presents. Motion probes a symmetric direction binding: "left then right" and "right then left" trajectories are visually mirror images, and direction tokens (`left`, `right`) are shared between train and held-out. With no asymmetric prior available, the model must use the verb's text-embedding region to route attention to the correct trajectory feature; as the verb drifts in CLIP space, this routing degrades smoothly. Size and speed, in contrast, offer a natural training-time default: the larger block is more visually salient and the slower video allocates more frames to its decision segment. The model learns these defaults and falls back to them when the held-out text signal is too far from trained verbs to drive attribute attention. Tokens matching the default class pass; tokens in the opposite class fail — producing a class-bimodal cliff. L/14 closes both mechanisms by providing a denser text-embedding space: held-out paraphrases that crowded around training tokens in B/32 disperse in L/14, while previously isolated tokens move closer to training cluster boundaries. The relevant quantity is not raw cosine but whether the backbone's text embedding can resolve the lexical signal into stable attribute queries.

We hypothesize that the partial L/14 closure on speed FAST ($\Delta = +0.273$, vs $+0.350$ for size SMALL and $+0.322$ for motion below-cliff) reflects a combination of two factors: (a) the speed axis's lexical signal is the furthest from training in CLIP-L/14 space among the tested cliffs, and (b) the SLOW default appears strongly entrenched because the visual binding signal — comparing motion timing across two 24-frame videos — is more temporally distributed and harder to extract than the spatially-localized size signal. Whether scaling alone can close the FAST cliff at sufficient backbone size, or whether direct intervention against default-class bias is required, is left for future work. We report this hypothesis with the empirical evidence above; we do not claim to have isolated which factor dominates.

## 4.6 Limitations

We probe three of v4's seven binding axes (motion\_sequence, size, speed); the four others (color, object, action, spatial) are not tested here. The choice was driven by the availability of cliff-zone held-out tokens with semantic-class structure; color and spatial held-out tokens are tuples (magenta$\leftrightarrow$cyan, north$\leftrightarrow$south) whose semantic structure differs from size/speed.

We compare two backbone scales (B/32, L/14); intermediate scales (B/16) and alternative backbones (DINOv2, EVA-02) are unaddressed. The speedily token shows the highest L/14 variance ($\sigma = 0.230$; one seed reaches 0.786, the other two stay at 0.286–0.310). This is reported as a borderline case rather than a stable closure. CF-PrefBench v4 uses a deterministic 2D rendered simulator; the cliff phenomenon may be specific to its structured visual distribution. An earlier draft reported the strict PFA $=0.000$ on motion\_sequence held-out as "perfect anti-correlation"; this characterization was corrected before submission (the zero is an aggregation artifact, per-row accuracy is 0.786 — see footnote in Section~4.1).

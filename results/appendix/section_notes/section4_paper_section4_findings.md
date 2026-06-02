# 4 Findings: The Lexical Cliff in Compositional Preference Learning

We report two empirical findings on CF-PrefBench v4. First, an instruction-conditioned binding module (LIB) trained on CLIP ViT-B/32 patch features exhibits a sharp accuracy drop when held-out paraphrases sit lexically far from the training pool. The drop reaches anti-correlation on the speed axis ("briskly" at 0.048 mean accuracy, worse than random chance). Second, this *lexical cliff* is not a property of the binding architecture: replacing the visual backbone with ViT-L/14 closes the cliff on two of three axes and reduces it on the third without any change to the dataset or training recipe. The cliff is a small-backbone capacity effect, not a universal CLIP limitation, and its mechanism differs across binding axes — cosine-monotonic on motion verbs, class-bimodal on size and speed.

## 4.1 Setup

**Benchmark.** CF-PrefBench v4 (built on the v3 dataset of \citet{cfprefbench-v3}) extends to 7 binding axes (color, object, action, spatial, size, motion\_sequence, speed) over 7{,}749 video-pair preference examples. Each example is a triple $(v_A, v_B, t)$ where the two videos differ on a single binding axis and the instruction $t$ resolves to a preferred video; counterfactual flip groups balance A and B labels exactly (fraction\_ab\_balanced = 1.0). Train, val, and test\_seen splits share paraphrase verbs; the test\_heldout\_lexical split substitutes synonymous verbs that never appear in training (e.g., motion train uses \{move, drag, push\} while held-out uses \{shift, convey, transit\}). Held-out paraphrase pools were verified disjoint from training (Appendix~B).

**Model.** LIB v0 is a per-attribute cross-attention module that conditions CLIP patch tokens on text via four learned attribute queries projected from the CLIP text embedding. It is trained with the four-term loss specified in Section~3 (BCE + recon + counterfactual + paraphrase stability), 60 epochs, AdamW (lr\_LIB$=$1e$-4$, lr\_head$=$5e$-4$), batch 32 (16 for ViT-L/14). All experiments below use three random seeds (1, 2, 3); reported numbers are seed means with standard deviations unless noted.

**Visual backbones.** We compare two CLIP visual encoders kept frozen during training: ViT-B/32 (a 768-dim, 49-patch encoder commonly used in compositional preference work) and ViT-L/14 (a 1024-dim, 256-patch encoder with $\sim$3$\times$ the parameters). Text embeddings are taken from the matching CLIP text encoder (512-dim for B/32; 768-dim for L/14).

**Anti-fabrication scope.** Every metric is computed over three seeds with the full per-row prediction rows preserved. We report both row-level accuracy and the strict-PFA aggregation; differences between these metrics are diagnosed in Section~4.2 rather than averaged away.

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

Accuracy decreases monotonically with the held-out verb's CLIP cosine to training. The cliff is gradual: scoot at cosine 0.927 lands at 0.679, equidistant between the success cluster (shift, convey at 0.92$\pm$) and the failure point (transit at 0.500). Pearson correlation over the 12 verb–seed observations is $r=0.808$ ($p=0.0015$); Spearman rank correlation is $\rho=0.880$ ($p=0.00016$). Figure~\ref{fig:motion-cliff-b32} plots the curve with per-seed dots.

\begin{figure}[t]
\centering
\includegraphics[width=0.48\columnwidth]{figures/fig_4_1_motion_cliff_b32.pdf}
\caption{Motion-verb cliff under ViT-B/32. Per-seed dots overlay
the 3-seed mean (error bars). Accuracy decreases monotonically with
CLIP cosine of the held-out verb to training.}
\label{fig:motion-cliff-b32}
\end{figure}

**Substitution isolates the failure to the text side.** To probe whether the cliff is caused by the held-out instruction text or by interaction with the video features, we re-run inference with the held-out instruction's CLIP text feature replaced by a training-verb instruction with identical direction tokens (e.g., "transit the block left then right" $\rightarrow$ "move the block left then right"). The video and label are unchanged. Table~\ref{tab:motion-subst} shows the resulting accuracy.

\begin{table}[t]
\centering
\small
\begin{tabular}{lrrr}
\toprule
Verb & Normal & Substituted & $\Delta$ \\
\midrule
shift & 0.917 & 0.929 & $+0.012$ \\
convey & 0.929 & 0.929 & $\pm 0.000$ \\
scoot & 0.679 & 0.917 & $+0.238$ \\
transit & 0.500 & 0.893 & $+0.393$ \\
\bottomrule
\end{tabular}
\caption{Substitution recovery on ViT-B/32 motion verbs. The text feature is swapped to a training-verb instruction; videos are unchanged. Recovery scales inversely with cosine, confirming the failure is text-side.}
\label{tab:motion-subst}
\end{table}

Recovery scales inversely with the held-out verb's cosine to training: verbs already inside the binding region (shift, convey) gain $\leq 0.01$; verbs in the cliff zone gain $+0.24$ to $+0.39$. The model can read direction correctly from the video; what fails on transit/scoot is the projection from a far-from-training text embedding to the right attribute queries.

**PFA versus per-row accuracy.** The flip-group preference accuracy (PFA) on motion\_sequence test\_heldout\_lexical is exactly 0.000 across all three seeds. This zero is an aggregation artifact: PFA requires \emph{every} paraphrase row in a flip group to be correct, and a single failing verb (transit) cascades the flip-group score to zero for all 14 groups. Per-row accuracy on the same predictions is 0.786, far from random. We retain the strict-PFA definition because flip-group consistency is a meaningful property, but report per-row accuracy alongside it whenever a single token dominates the error distribution.

## 4.3 Cross-Architecture Replication: ViT-L/14 Closes the Motion Cliff

If the cliff stems from the small text-embedding capacity of B/32, a larger CLIP variant should reduce or remove it. We replicate Section~4.2's setup with ViT-L/14 (1024-dim, 256-patch). Visual features are re-extracted for all 1{,}080 new-axis videos; LIB v0 is re-trained from scratch for three seeds with the same recipe (epoch budget unchanged; batch reduced to 16 for memory). Table~\ref{tab:motion-l14} compares the cliff across architectures.

\begin{table}[t]
\centering
\small
\begin{tabular}{lrrrrr}
\toprule
Verb & cos(B/32) & B/32 acc & cos(L/14) & L/14 acc & $\Delta$ \\
\midrule
shift & 0.968 & 0.917 & 0.925 & 0.905 & $-0.012$ \\
convey & 0.938 & 0.929 & 0.918 & 0.929 & $\pm 0.000$ \\
scoot & 0.927 & 0.679 & 0.875 & 0.905 & $+0.226$ \\
transit & 0.915 & 0.500 & 0.876 & 0.917 & $+0.417$ \\
\bottomrule
\end{tabular}
\caption{Motion-verb cliff under two backbones. The two verbs that cliff on B/32 (transit, scoot) recover to $\geq 0.905$ accuracy under L/14, even though their L/14 cosines to training are \emph{lower} (0.876, 0.875). The cliff is backbone-specific, not a universal CLIP property.}
\label{tab:motion-l14}
\end{table}

The two B/32-cliff verbs gain $+0.226$ (scoot) and $+0.417$ (transit) absolute under L/14. Crucially, this happens \emph{despite} L/14 placing both verbs at lower cosine to training than they were on B/32. There is no universal cosine threshold for direction binding; the threshold is determined by the backbone's text-embedding capacity. The substitution experiment becomes uninformative on L/14 (the held-out verbs already pass), consistent with the failure mode being absent rather than masked. Figure~\ref{fig:motion-dual-arch} overlays the two backbone curves.

\begin{figure}[t]
\centering
\includegraphics[width=0.48\columnwidth]{figures/fig_4_2_motion_dual_arch.pdf}
\caption{Motion-verb cliff under two backbones. ViT-L/14 (red) lifts
the two cliff verbs (transit, scoot) to $\geq 0.905$ accuracy
even though their L/14 cosines are lower than under ViT-B/32.}
\label{fig:motion-dual-arch}
\end{figure}

## 4.4 Axis Generalization: Two Distinct Mechanisms

To test whether the cliff generalizes across binding axes, we evaluate the same six trained models (3 B/32 seeds, 3 L/14 seeds) on eight new held-out tokens spanning the size and speed axes: \{colossal, gigantic\} (BIG class), \{miniature, petite\} (SMALL class), \{briskly, speedily\} (FAST class), and \{sluggishly, gradually\} (SLOW class). New tokens are chosen to span the CLIP-B/32 cliff zone $[0.89, 0.95]$ while staying inside the original training axis's semantic class. No retraining is needed: the v4 training data excludes all of these tokens by construction, so the existing checkpoints are a clean held-out evaluation. Table~\ref{tab:3axis-cliff} aggregates results by semantic class.

\begin{table}[t]
\centering
\small
\begin{tabular}{llrrr}
\toprule
Axis & Class & B/32 acc & L/14 acc & $\Delta$ \\
\midrule
\multirow{2}{*}{Size}
 & BIG (colossal, gigantic) & 0.925 $\pm$ 0.009 & 0.810 $\pm$ 0.067 & $-0.115$ \\
 & SMALL (miniature, petite) & \textbf{0.444} $\pm$ 0.103 & 0.794 $\pm$ 0.095 & $+0.350$ \\
\multirow{2}{*}{Speed}
 & FAST (briskly, speedily) & \textbf{0.115} $\pm$ 0.084 & 0.389 $\pm$ 0.179 & $+0.273$ \\
 & SLOW (sluggishly, gradually) & 0.829 $\pm$ 0.134 & 0.996 $\pm$ 0.009 & $+0.167$ \\
\multirow{2}{*}{Motion}
 & Above-cliff (shift, convey) & 0.923 $\pm$ 0.009 & 0.917 $\pm$ 0.017 & $-0.006$ \\
 & Below-cliff (scoot, transit) & \textbf{0.589} $\pm$ 0.107 & 0.911 $\pm$ 0.014 & $+0.322$ \\
\bottomrule
\end{tabular}
\caption{Three-axis $\times$ two-architecture cliff matrix. The cliff appears on all three axes under B/32 (boldface = cliff cells). ViT-L/14 fully closes the cliff on motion and size; on speed, the FAST class only partially recovers (0.389 mean, still well below ceiling). $n=6$ per cell (2 tokens $\times$ 3 seeds).}
\label{tab:3axis-cliff}
\end{table}

Two patterns emerge.

**Pattern 1 (motion): cosine-monotonic.** On motion verbs, accuracy is a smooth function of CLIP cosine to training (Section~4.2 Pearson $r=0.808$). Below-cliff verbs (transit at cos 0.915, scoot at 0.927) drop continuously from the success plateau.

**Pattern 2 (size, speed): class-bimodal default bias.** On size and speed, held-out tokens within the same semantic class as training tokens have nearly-identical CLIP cosines (size SMALL: 0.917 vs 0.918; speed FAST: 0.937 vs 0.942), yet split into two regimes. One regime (size BIG, speed SLOW) succeeds at 0.83–0.93; the other (size SMALL, speed FAST) collapses to 0.04–0.49. The accuracy of size SMALL and speed FAST is not predicted by cosine — both classes' tokens cluster at the same cosine — but is fully determined by class membership. Speed briskly at 0.048 is below chance, indicating systematic anti-correlation: the model confidently picks the slower video even when told "move the block briskly."

The asymmetry suggests B/32-trained LIB has learned a **default-class bias**: prefer the larger block; prefer the slower video. Held-out tokens that happen to fall in the default class succeed because the default is correct for them; held-out tokens in the opposite class fail because the held-out lexical signal cannot override the default. The motion axis lacks an obvious default (direction tokens are symmetric in the training distribution), so its cliff degrades smoothly with cosine instead of snapping to a default.

**L/14 closes both mechanisms with one exception.** L/14 closes the motion cliff fully and the size cliff fully (SMALL class lifts 0.444 $\rightarrow$ 0.794), even though the L/14 cosine for SMALL-class tokens is $\sim$0.875. On the speed axis, L/14 closes the SLOW class to ceiling (0.996) but only partially lifts FAST (0.115 $\rightarrow$ 0.389; briskly stays at 0.317, speedily reaches 0.460 with one seed at 0.786). Scaling helps, but not uniformly; the SLOW default appears strong enough at training that even L/14 cannot fully override it for held-out FAST tokens. Figure~\ref{fig:3axis-cliff} visualizes the matrix; Figure~\ref{fig:mechanism} contrasts the two mechanisms.

\begin{figure*}[t]
\centering
\includegraphics[width=0.95\textwidth]{figures/fig_4_3_3axis_matrix.pdf}
\caption{Three-axis $\times$ two-architecture cliff matrix
($n=6$ per cell). Bold ``CLIFF'' arrows mark cells where the
B/32 model drops to or below chance. ViT-L/14 fully closes the cliff
on motion and size; on the speed axis it only partially closes the
FAST class.}
\label{fig:3axis-cliff}
\end{figure*}

\begin{figure}[t]
\centering
\includegraphics[width=0.95\columnwidth]{figures/fig_4_4_mechanism_diagram.pdf}
\caption{Two cliff mechanisms. (a) Motion verbs: cliff is a smooth
sigmoid in held-out CLIP cosine to training. (b) Size and speed:
held-out tokens in the same semantic class as training cluster at
the same cosine, but accuracy splits along a learned default-class
prior (BIG/SLOW passes, SMALL/FAST fails).}
\label{fig:mechanism}
\end{figure}

## 4.5 Mechanism Discussion

We interpret the two cliff mechanisms (cosine-monotonic on motion, class-bimodal on size and speed) as consequences of the visual-temporal structure each axis presents.

**Motion verbs probe a symmetric direction binding.** "Left then right" and "right then left" trajectories are visually identical up to a reflection, and the model must read direction tokens to disambiguate. Direction tokens (`left`, `right`) are shared across train and held-out instructions; only the verb changes. Because no asymmetric prior is available, the model must use the verb's text-embedding region to route attention to the correct trajectory feature. As the held-out verb drifts away from training in CLIP space, this routing degrades smoothly — hence the monotonic cliff.

**Size and speed offer an asymmetric prior.** Both axes have a natural default in the training distribution: the larger block is more visually salient and more frequently the target; the slower video allocates more frames to the motion segment that decides the label. The model learns this default at training time and uses it as a fallback when the held-out text signal is too far from the trained verbs to drive attribute attention. Held-out tokens that match the default class (BIG, SLOW) get correctly identified by the default; held-out tokens in the opposite class (SMALL, FAST) fail because the default beats the weak lexical signal. This produces a class-bimodal cliff rather than a smooth cosine slope.

**Why L/14 closes the cliff.** A larger CLIP backbone has a denser, more semantically organized text embedding space \citep{radford2021clip}. Held-out paraphrases that crowd around training tokens in B/32 disperse in L/14, while previously isolated tokens (transit, scoot) move closer to training cluster boundaries. This is consistent with our cosine measurements: shift cosine drops from 0.968 (B/32) to 0.925 (L/14), and transit drops from 0.915 to 0.876, yet L/14 accuracy on transit jumps from 0.500 to 0.917. The relevant quantity is not raw cosine but whether the backbone's text embedding can resolve the lexical signal into stable attribute queries — which L/14's larger embedding evidently can.

**Why speed FAST is the hardest case.** L/14 closes the size SMALL cliff fully (0.444 $\rightarrow$ 0.794) but only partially closes speed FAST (0.115 $\rightarrow$ 0.389). One possible explanation: the speed axis's binding signal in video is temporally distributed — the model must compare the timing of motion across two 24-frame videos — while the size signal is spatial and visible in a single frame. The default-class bias may be more entrenched on the temporal axis because the visual evidence is harder to extract. This implies that backbone scaling alone is insufficient when both the lexical and visual binding signals are degraded relative to training.

## 4.6 Limitations

**Three of seven axes tested.** We probe the cliff on motion\_sequence, size, and speed. The four other axes in v4 (color, object, action, spatial) are not tested here; whether they exhibit cliffs is left to future work. The choice of these three axes was driven by the availability of cliff-zone held-out tokens with semantic-class structure; for color and spatial, the held-out tokens are tuples (magenta$\leftrightarrow$cyan, north$\leftrightarrow$south) whose semantic class structure differs from size/speed.

**One model architecture per scale.** We compare ViT-B/32 and ViT-L/14 specifically. We do not test intermediate scales (e.g., ViT-B/16) or alternative backbones (DINOv2, EVA-02). Whether the cliff transition with backbone scale is gradual or sharp is an open question; a fuller scaling study would require six or more CLIP variants.

**Per-token variance on speed.** The speedily token shows the highest L/14 variance ($\sigma = 0.230$; one of three seeds reaches 0.786, the other two stay at 0.286–0.310). This single-seed effect is not stable enough to claim the cliff is closeable for FAST class with more seeds, but it does suggest that briskly's failure may be more entrenched than speedily's.

**Domain.** CF-PrefBench v4 uses a deterministic 2D rendered simulator; the cliff phenomenon may be specific to the structured, low-variance visual distribution of this benchmark. Replication on a natural-video preference dataset would strengthen the generality claim.

**The 0.000 PFA mischaracterization.** An earlier draft of this work reported the PFA $=0.000$ on motion\_sequence held-out as "perfect anti-correlation." This is incorrect — as shown in Section~4.2, the per-row accuracy is 0.786, and the zero reflects strict-PFA aggregation cascading a single failing verb. We retain the strict PFA metric definition but cross-check with per-row accuracy whenever it diverges; the original "anti-correlation" framing was corrected before this submission.



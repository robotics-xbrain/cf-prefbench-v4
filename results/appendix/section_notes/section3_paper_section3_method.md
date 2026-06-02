# 3 CF-PrefBench v4 and the LIB v0 Method

## 3.1 Task Formulation

We study compositional preference learning over short trajectory pairs. An example is a triple $(v_A, v_B, t)$ where $v_A$ and $v_B$ are two video trajectories that differ on a single \emph{binding axis} (e.g., the color, size, or motion direction of a target object) and $t$ is a natural-language instruction. The task is to predict the preferred trajectory $y \in \{A, B\}$ — the one whose state evolution matches the instruction. Compositional binding is the model's task of resolving \emph{which} attribute the instruction references (axis identification) and \emph{which} of the two trajectories satisfies it (preference selection). Following \citet{cfprefbench-v3}, we measure performance by per-row row-level accuracy and flip-group preference accuracy (PFA), the fraction of counterfactual flip groups whose paraphrase rows are all correctly predicted.

The critical evaluation split is the \emph{held-out lexical} split: instructions for held-out test examples are paraphrased with verbs that never appear in training. This is the split on which Section~4's lexical cliff emerges and where Section~5's sanity battery is most informative. A model that succeeds on train and test\_seen but fails on test\_heldout\_lexical has memorized the training-verb-to-binding mapping rather than learned the binding itself. Figure~\ref{fig:benchmark-schematic} illustrates a single counterfactual flip group from the motion\_sequence axis.

\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig_3_1_benchmark_schematic.pdf}
\caption{CF-PrefBench v4 task formulation. Each example is a triple
$(v_A, v_B, t) \rightarrow $ preferred $\in \{A, B\}$. Counterfactual
flip groups share the same two videos but pair them with paraphrased
instructions that flip the preferred label; a compositional model
should flip its preference when the instruction flips. The
test\_heldout\_lexical split uses paraphrase verbs that never appear
in training (held-out pool in Table~\ref{tab:paraphrase-pools}); a
model that memorizes training-verb-to-binding mappings instead of
learning binding will fail on this split, as Section~4 demonstrates.}
\label{fig:benchmark-schematic}
\end{figure*}

## 3.2 CF-PrefBench v4 Benchmark Design

CF-PrefBench v4 builds on the v3 dataset of \citet{cfprefbench-v3} and adds three new binding axes (size, motion\_sequence, speed) using a deterministic 2D OpenCV-rendered simulator. The full benchmark covers seven binding axes — color, object, action, spatial, size, motion\_sequence, speed — plus an impossible\_premise diagnostic split. The benchmark contains 7{,}749 video-pair preference examples (Table~\ref{tab:v4-splits}) drawn from 2{,}646 unique 24-frame, $192 \times 144$ rendered videos.

\begin{table}[t]
\centering
\small
\begin{tabular}{lrrrrrrrrr}
\toprule
 & train & val & test\_seen & lex & cam & color & spatial & hard & total \\
\midrule
color & 528 & 84 & 72 & 84 & 72 & 84 & 84 & 72 & 1080 \\
object & 528 & 84 & 72 & 84 & 72 & 84 & 84 & 72 & 1080 \\
action & 528 & 84 & 72 & 84 & 72 & 84 & 84 & 72 & 1080 \\
spatial & 528 & 84 & 72 & 84 & 72 & 84 & 84 & 72 & 1080 \\
size & 528 & 84 & 72 & 84 & 72 & 84 & 84 & 72 & 1080 \\
motion\_seq. & 528 & 84 & 72 & 84 & 72 & 84 & 84 & 72 & 1080 \\
speed & 528 & 84 & 72 & 84 & 72 & 84 & 84 & 72 & 1080 \\
imposs.\_prem. & 0 & 27 & 27 & 27 & 27 & 27 & 27 & 27 & 189 \\
\midrule
total & 3{,}696 & 615 & 531 & 615 & 531 & 615 & 615 & 531 & 7{,}749 \\
\bottomrule
\end{tabular}
\caption{CF-PrefBench v4 split sizes by axis. ``lex'', ``cam'', ``color'', ``spatial'', ``hard'' are the five test splits: test\_heldout\_lexical, test\_heldout\_camera, test\_heldout\_color, test\_heldout\_spatial, test\_hard\_negatives. The impossible\_premise axis is a diagnostic split with no training examples by design; it is held out at train time and only evaluated at test.}
\label{tab:v4-splits}
\end{table}

\textbf{Counterfactual flip groups.} Each axis has 88 train flip groups, where a flip group is a pair of trajectory videos that share scene composition but differ only on the binding axis. The same two videos appear in the flip group with both paraphrase variants of the instruction (e.g., ``pick up the large red block'' and ``pick up the small red block''); a perfectly compositional model should flip its preference when the instruction flips. PFA is computed at the flip-group level; row-level accuracy at the per-row level.

\textbf{Paraphrase pools.} For each axis we curate two disjoint pools of paraphrase verbs, one for training and one for the held-out lexical test split. Pools are listed in Table~\ref{tab:paraphrase-pools}. The pools were chosen so that held-out paraphrases span the CLIP-text cosine cliff zone (cosine $0.87$–$0.97$ to training-verb instructions on ViT-B/32). Section~4 quantifies the resulting cliff curve and Section~5 verifies that the train/held-out paraphrase split is clean.

\begin{table}[t]
\centering
\small
\begin{tabular}{lll}
\toprule
Axis & Training paraphrases & Held-out paraphrases \\
\midrule
color & activate, press, touch & engage, tap, trigger \\
object & bring, move, place & carry, deliver, transport \\
action & close, move, open, pull, push & extract, operate, retract, slide \\
spatial & move, place, put & drift, shift, transfer \\
size & grasp, lift, pick & fetch, retrieve, secure \\
motion\_seq. & drag, move, push & convey, shift, transit (+ scoot, Exp 3) \\
speed & carry, move, transport & advance, shift, translate \\
\bottomrule
\end{tabular}
\caption{Train and held-out paraphrase pools per binding axis. Pools are disjoint by construction; the held-out pool is used only in the test\_heldout\_lexical split. For motion\_sequence, we added a fourth held-out verb ``scoot'' in Section~4 to probe the cliff transition zone.}
\label{tab:paraphrase-pools}
\end{table}

\textbf{Trajectory generation.} Videos are rendered by a deterministic 2D simulator in OpenCV at $192 \times 144$ resolution, 24 frames per video. The simulator places named primitive shapes (block, puck, ball) at instructed positions and animates them along scripted trajectories that resolve the binding signal (e.g., the target block moves to a goal while the distractor stays still). The labeling is derived deterministically from the simulator state, not from human judgment; the IAA question is therefore not applicable and we flag it as a domain limitation in Section~6.4.

## 3.3 Evaluation Protocol

Each method is evaluated on val, test\_seen, and the five held-out splits. We report row-level accuracy as the primary metric and PFA as a secondary metric; the two diverge when a single failing paraphrase token cascades the strict PFA aggregation to zero (footnote in Section~4.1). For all learned methods we train three random seeds (1, 2, 3) and report seed-means with standard deviations. For zero-shot VLM baselines (Section~3.5) we run a single deterministic pass with temperature $0$. Section~4 introduces cosine-similarity-stratified analysis on top of these standard metrics: for each held-out lexical token we compute the CLIP-text cosine similarity to the mean training-verb instruction and report accuracy as a function of this cosine.

## 3.4 The LIB v0 Architecture

LIB v0 (Learned Instruction-conditioned Binding) is a per-attribute cross-attention module that conditions CLIP patch tokens on the instruction text. Given a video, we sample $K{=}8$ frames uniformly across the 24-frame sequence and encode each frame with a frozen CLIP visual encoder, producing $K \times P$ patch tokens per video (P=49 for ViT-B/32, P=256 for ViT-L/14). The instruction is encoded by the matching CLIP text encoder.

\begin{itemize}
\item \textbf{Per-attribute queries.} A linear projection $q_\text{proj}: \mathbb{R}^{d_\text{text}} \to \mathbb{R}^{n_\text{attr} \times d_\text{attr}}$ produces $n_\text{attr}{=}4$ attribute-conditioned queries from the CLIP text embedding ($d_\text{attr}{=}128$).
\item \textbf{Cross-attention.} The queries attend over the flattened $K \cdot P$ patch tokens via a 4-head multi-head attention layer, followed by LayerNorm and residual connection.
\item \textbf{Binding score.} A second projection $q_\text{exp}: \mathbb{R}^{d_\text{text}} \to \mathbb{R}^{n_\text{attr} \times d_\text{attr}}$ produces the ``expected'' per-attribute embeddings; the cosine similarity between attended embedding and expected embedding gives a per-attribute binding score in $[-1, 1]$.
\item \textbf{Preference head.} The 4-d binding vectors for the two videos plus a small text projection (16-d) are concatenated and passed through a 64-hidden-unit MLP to produce a scalar preference score.
\item \textbf{Reconstruction auxiliary.} A small per-axis classifier head over the attended embeddings reconstructs the instruction's attribute (color, object, action, spatial), preventing the attribute queries from collapsing to a degenerate solution.
\end{itemize}

Total trainable parameters: $\sim$0.6M (ViT-B/32 configuration), $\sim$0.9M (ViT-L/14, due to larger projection dimensions). The frozen CLIP visual encoder ($\sim$87M params for ViT-B/32, $\sim$305M for ViT-L/14) is not counted.

\textbf{Training.} AdamW with lr$_\text{LIB}{=}1\mathrm{e}{-4}$, lr$_\text{head}{=}5\mathrm{e}{-4}$, weight decay $1\mathrm{e}{-4}$. Four-term loss combining preference BCE (weight 1.0), reconstruction CE (0.1), counterfactual margin (0.05), and paraphrase stability (0.02). Three seeds, 60 epochs, batch size 32 for B/32 and 16 for L/14 (to fit in 24~GB).

## 3.5 Baselines

We compare four classes of method.

\textbf{Engineered centroid.} A 173-dim hand-engineered feature per trajectory pair: instruction-conditioned color-centroid pixel statistics, frame-averaged motion deltas, and a bag-of-words representation of the instruction. A 2-layer MLP preference head is trained on this feature. The baseline reached the 0.929 ($=13/14$) PFA ceiling on v3 test\_heldout\_color color-axis with eight CPL loss variants $\times$ three seeds in our prior work, and is included here as a strong engineered reference \citep{cfprefbench-v3}.

\textbf{LIB v0 (B/32 and L/14).} The architecture in Section~3.4 on two CLIP backbones. The B/32 configuration is the original Phase 1 model; the L/14 configuration is the Section~4.3 scaling baseline.

\textbf{Phase 3 hybrid.} A learned-gate blend of LIB v0 and the engineered centroid pathway. The hybrid was our own architecture for Phase 3 and is included here both as a historical baseline and as the canonical example for the sanity battery in Section~5 (where Section~5.3 shows the hybrid is a centroid pass-through).

\textbf{Zero-shot VLM judges: GPT-4o and Qwen2-VL-2B.} For each preference example we sample $K{=}8$ frames per video, lay them out as a $2 \times 4$ image grid composite per trajectory, and prompt the VLM with the two images and the instruction. We extract the answer A/B/Tie from a strict-JSON response (full prompt in Appendix~D). GPT-4o-2024-11-20 is run via the OpenAI API at temperature 0; Qwen2-VL-2B is run locally on a single GPU at temperature 0. Qwen2-VL-2B is a substitute for the originally-planned Qwen2.5-VL-7B because the 7B checkpoint was not locally available and a 14~GB proxy download was infeasible in our environment; the 2B substitute serves as a capability-floor baseline (Section~4.4).

## 3.6 The Sanity Diagnostic Battery (forward reference)

Section~5 introduces a six-probe sanity battery (SC-1..SC-6) for compositional preference methods. Probes target three failure modes: pass-through pathways in hybrid architectures (SC-1..SC-3, SC-4), feature redundancy across pathways (SC-5), and instruction non-use (SC-6). The full probe descriptions, applicability matrix, and case studies are deferred to Section~5; here we note that LIB v0 trains and evaluates use only the architectural components defined in Section~3.4, and all sanity probes are computed post-hoc on the trained model without modifying training.

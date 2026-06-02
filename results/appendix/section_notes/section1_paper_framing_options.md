# Section 1 (Introduction) — Three Candidate Framings

**Status**: OPTIONS ONLY. No draft committed. User picks one tomorrow when fresh.

The paper has two distinct contributions and three plausible ways to lead with them:

- **Framing 1 (Cliff-First)** — leads with the empirical phenomenon. Best for ACL/EMNLP main reviewers who reward novel empirical findings.
- **Framing 2 (Diagnostic-First)** — leads with the sanity battery as methodology. Best for evaluation/methodology-focused reviewers.
- **Framing 3 (Dual-Contribution)** — balances the two. Best when the audience composition is unknown.

Below: a one-paragraph hook, contributions bullets, and a one-paragraph outline for each. All three can be written in ~500 words for the final intro; this document is a sketch for the user to decide on.

---

## Framing 1: Cliff-First

### Hook (1 paragraph)

Compositional preference methods are typically evaluated on held-out trajectory pairs but with paraphrased instructions drawn from the same training-time verb pool. We construct a benchmark where the held-out paraphrase verbs are deliberately disjoint from training — \{shift, convey, transit, scoot\} replacing \{move, drag, push\} on a motion-direction task — and find that a CLIP-ViT-B/32-based binding model has a sharp accuracy cliff in held-out verb cosine similarity to training: above cosine $0.94$ accuracy is near ceiling ($\geq 0.92$); below cosine $0.92$ it falls to chance or below. Scaling to ViT-L/14 closes the cliff on motion and size axes but only partially on speed; zero-shot GPT-4o handles the size and speed cliff tokens correctly but cannot read motion direction at $192 \times 144$ resolution. The cliff is not a property of vision-language models in general; it is a property of how a specific binding head couples to a specific text-embedding capacity.

### Contributions (bullets)

- We document a CLIP-text-cosine lexical cliff in compositional preference learning, with reproducible $r{=}0.808, p{=}0.0015$ (Pearson, $n{=}12$ verb-seed observations) on motion verbs and class-bimodal default-bias variants on size and speed.
- We show the cliff is backbone-specific: ViT-L/14 fully closes it on motion and size, partially closes it on speed FAST. We extend the analysis to three method classes (engineered baseline, LIB v0 at two scales, hybrid, two zero-shot VLMs) and find that no single method dominates the cliff matrix across axes.
- We extend CF-PrefBench to v4 with three new binding axes (size, motion\_sequence, speed) and held-out paraphrase pools chosen to span the cliff zone. v4 is the first compositional preference benchmark to ship with cosine-stratified evaluation built in.
- We present a six-probe sanity battery that caught a centroid pass-through in our own prior architecture and falsifies four anti-collapse mechanisms (Section~5).

### Outline (1 paragraph)

Section~2 reviews related work on compositional preference learning, CLIP lexical generalization, and VLM-as-judge evaluation. Section~3 introduces CF-PrefBench v4 and the LIB v0 method. Section~4 reports the lexical cliff on motion (4.2), demonstrates that L/14 closes it (4.3), generalizes the finding to size and speed (4.4), and presents the cross-method comparison (4.4, Figure~\ref{fig:cross-method-cliff}). Section~5 introduces the six-probe sanity battery as a methodological contribution and validates it on three case studies including the cliff results themselves. Section~6 discusses implications, limitations, and future work.

### Audience

ACL/EMNLP main reviewers who weight novel empirical findings. The hook is dramatic (a cliff!) but defensible (we report exactly when and why it appears). Risk: a reviewer who reads "the cliff" as a finding-of-finding-driven paper may underweight the methodological battery in Section 5.

---

## Framing 2: Diagnostic-First

### Hook (1 paragraph)

We report a near-miss false positive: in a prior version of this work we trained a hybrid CLIP-attention + engineered-feature preference model that scored $0.905 \pm 0.034$ on a held-out color-binding task, near the engineered ceiling. The headline result was a positive method. A six-probe sanity battery applied post-hoc revealed that the learned attention contributed nothing to the prediction — zeroing the learned pathway left accuracy at $0.929$, and zeroing the engineered pathway dropped accuracy to $0.167$. The hybrid was a centroid pass-through with a decorative attention module that the head had learned to ignore. This is a representative failure mode for compositional preference methods that combine learned and engineered signal pathways, and standard accuracy-based evaluation cannot distinguish it from a real positive result. We present the sanity battery as a reusable diagnostic and apply it to a phenomenon, the lexical cliff, that the battery also helped us correctly characterize.

### Contributions (bullets)

- We introduce a six-probe sanity battery for compositional preference methods. The battery converts noisy NORMAL-accuracy rankings into unambiguous falsifications of three failure modes: pass-through pathways in hybrid architectures, gate triviality, and instruction non-use.
- We apply the battery in three case studies: detecting centroid pass-through in our own prior hybrid (Section~5.3), validating that the lexical cliff finding is text-side (Section~5.4), and falsifying four anti-collapse architectures designed to fix the pass-through (Section~5.5).
- We document a lexical cliff in compositional preference learning, with the battery confirming that the cliff is text-encoding-side and that scaling to ViT-L/14 closes it on motion and size axes but not speed (Section~4).
- We release CF-PrefBench v4 with held-out lexical splits that surface the cliff and built-in cosine-stratified evaluation that supports sanity-battery-style analysis (Section~3).

### Outline (1 paragraph)

Section~2 reviews related work on compositional preference, CLIP sanity, and VLM evaluation. Section~3 introduces CF-PrefBench v4, the LIB v0 method, and the methods we compare. Section~4 reports the lexical cliff phenomenon empirically as the running test case for the battery. Section~5 introduces the six-probe sanity battery, presents its design rationale, and walks through three case studies; Section~5.3 is the centroid pass-through; Section~5.4 is the cliff text-side validation; Section~5.5 is the anti-collapse falsification. Section~6 discusses implications, limitations, and future work.

### Audience

Evaluation and methodology-focused reviewers who weight reusable tools. The "we caught a near-miss false positive" hook is honest and memorable; reviewers value methodology that prevents publication errors. Risk: a reviewer who reads "the sanity battery is the contribution" may underweight the cliff finding as a stand-alone phenomenon.

---

## Framing 3: Dual-Contribution

### Hook (1 paragraph)

We make two contributions to compositional preference learning: a reproducible empirical phenomenon and a reusable diagnostic tool that produced it. The phenomenon is a lexical cliff — held-out paraphrases drift in CLIP-text-cosine to training, and below a backbone-specific cosine threshold a binding model's accuracy collapses (motion verbs Pearson $r{=}0.808$, $n{=}12$ verb-seed observations; size and speed show a related class-bimodal default-bias variant). The tool is a six-probe sanity battery that we designed after a self-caught false positive on our own architecture and that caught both the centroid pass-through and the original mischaracterization of the cliff as ``perfect anti-correlation.'' We pair the two contributions because their development was intertwined: the battery exposed the cliff's text-side mechanism, and the cliff results provide the third case study that validates the battery beyond our own prior hybrid.

### Contributions (bullets)

- A lexical cliff in compositional preference learning, with two mechanisms (motion: smooth cosine; size/speed: class-bimodal default bias) and an architecture-scaling story (L/14 closes motion+size, partially closes speed FAST). Cross-method comparison shows the cliff is method-specific to LIB+B/32, not universal.
- A six-probe sanity battery for compositional preference methods. Three case studies (centroid pass-through, cliff text-side validation, anti-collapse falsification) on three different architecture classes.
- CF-PrefBench v4: 7 binding axes, 7749 examples, held-out paraphrase pools chosen to span the cliff zone, and cosine-stratified evaluation as standard reporting.
- An honest account of how the battery and the phenomenon co-evolved, including the post-hoc design of the battery after a near-miss false positive.

### Outline (1 paragraph)

Section~2 reviews related work. Section~3 introduces CF-PrefBench v4, LIB v0, baselines, and the sanity-battery protocol. Section~4 reports the cliff empirically; Section~5 presents the battery and its three case studies, with Section~5.4 explicitly using the Section~4 results as one of the validation case studies; Section~6 discusses the two contributions jointly. The dual contribution is reflected in the Section ordering: empirical finding (4) follows methodological infrastructure (3) and precedes diagnostic methodology (5), so each builds on the previous.

### Audience

Both communities. The "dual contribution" framing acknowledges that the paper has two distinct things to say and that they are tightly coupled. Risk: dual-contribution papers can read as unfocused; the hook needs to make the coupling concrete (we do this with "the battery and the cliff co-evolved").

---

## How to Pick (recommendations to the user)

Three considerations:

1. **Reviewer mix.** EMNLP main has both empirical-finding and methodology reviewers. Framing 3 (Dual) is the safest bet; Framing 1 (Cliff-First) is the riskiest but highest-ceiling.

2. **Paper title alignment.** Whichever framing you pick should reflect the title:
   - Framing 1 title pattern: ``The Lexical Cliff in Compositional Preference Learning''
   - Framing 2 title pattern: ``A Sanity Battery for Compositional Preference Methods'' (subtitle: ``With a Lexical Cliff Case Study'')
   - Framing 3 title pattern: ``CF-PrefBench v4: A Benchmark, a Lexical Cliff, and a Sanity Battery'' (long but covers both)

3. **What the body actually does.** The current Sections 4 and 5 are roughly equally weighted in length. Framing 3 matches that balance; Framing 1 will require trimming Section 5 mentions in the intro; Framing 2 will require trimming Section 4 mentions in the intro. None requires re-writing the body.

The current honest read of the paper, in my view (Claude's): **Framing 3 (Dual-Contribution) is the most defensible** given the body's balance, and the "the battery and the cliff co-evolved" hook is genuinely interesting because it's a self-correcting research narrative. But Framing 1 (Cliff-First) gives the strongest single sentence to lead with, which matters for reviewer fatigue. Framing 2 (Diagnostic-First) is the safest bet if the paper has to compete with stronger empirical results in its review batch — methodology contributions are easier to defend against "is this novel enough" critiques.

**This is for the user to decide.** All three framings are written from the same evidence base; the choice is rhetorical, not factual.

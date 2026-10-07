# ASD-STE100 compliance review

ASD-STE100 (Simplified Technical English) is a controlled-language standard
written for aircraft maintenance and other safety-critical procedural
documentation. Its own scope statement limits it to procedural and
descriptive technical writing — installation steps, operating instructions,
fault-isolation text. It explicitly does not cover contractual, marketing,
or argumentative writing, because its ~875-word approved dictionary and
single-meaning-per-word rule cannot express that kind of content without
deleting it.

This review applies the real rule set, not a loose paraphrase of it, and
reports actual findings from the text in this repository, not invented
examples.

## Scope: which files the standard actually applies to

| file | genre | in scope? |
|---|---|---|
| `README.md` §7 "Running it yourself", the `obsidian-vault/Running the Code.md` note, `tutorial.html`'s run block | installation and operating instructions | **yes** — this is exactly what STE is for |
| `README.md` §1–§6 (the SOC explainer, the mapping, the findings, the philosophy summary), `ANALYSIS.md`, `PHILOSOPHY.md`, `NOVELTY_RESULTS.md`, and 20 of the 24 vault notes | argumentative and explanatory prose, citing a technical and philosophical literature | **no**, by the standard's own design |

The three install/run documents say the same thing in slightly different
words. They are audited together below as one body of text.

## Part 1 — the procedural text (in scope)

Findings from `obsidian-vault/Running the Code.md`, with the specific STE
rule each line breaks.

| location | text | rule broken | fix |
|---|---|---|---|
| line 7 | "Reference implementation: Agents.jl, cross-checked against a dependency-free Python port..." | not a sentence (no subject + finite verb); passive "cross-checked" | "The Julia package Agents.jl is the reference implementation. We checked it against a Python port that uses no external packages." |
| lines 14–16, 20–23, 27–29 | a code block with no instruction sentence before it | every action in a procedure must have an explicit imperative sentence | add "Type this command:" (or equivalent) before each block |
| line 31 | heading "Run it" | "it" has no stated antecedent in the heading itself | "Run the Simulation" |
| line 34 | `# BTW control, all five drive rules, novelty sweep, finite-size scaling` | noun-phrase fragment, four-item noun cluster, no verb | split into one labelled command per behavior, each with its own one-line instruction |
| line 35 | `~1-2 min` | STE requires words, not symbols, for approximation | "about 1 to 2 minutes" |
| line 38 | "a video of the field evolving (real MP4, via Makie's bundled ffmpeg)" | `-ing` used as an adjective; parenthetical aside inside an instruction line | "a video that shows how the field changes. The script makes an MP4 file. It uses the ffmpeg tool that comes with Makie." |
| line 50 | "`make_video.jl` is reviewed, not run" | passive voice | "We reviewed `make_video.jl`. We did not run it." |
| line 52 | "If it errors on `agent_step!`/`model_step!` arguments" | "errors" used as a verb is not standard English and is not in the approved dictionary | "If the program shows an error about the arguments `agent_step!` or `model_step!`," |
| line 53–55 | "...run `?abmvideo` in the Julia REPL first — the fix is almost certainly a small keyword-name tweak, not a rewrite. `run_demo.jl` is lower risk: it's a closer port..." | one sentence joined by an em-dash (25 words, limit is 20 for an instruction); contraction "it's" is forbidden; "tweak" is informal vocabulary; "almost certainly" is an unapproved hedge | "Enter the command `?abmvideo` in the Julia command line. Read the result. The fix is probably a small change to a keyword name. A full rewrite is probably not necessary. The script `run_demo.jl` has less risk of failure. It stays closer to the Python version that we already tested." |
| line 57 | "No Julia install needed at all to explore the model interactively" | sentence fragment, no finite verb; "at all" is an informal intensifier | "You do not need to install Julia to use the model. Open the Interactive Tutorial. It runs in a web browser." |
| line 61 | "...for what the numbers these scripts print actually mean" | a wh-clause nested inside a wh-clause; "actually" is a filler adverb | "These notes explain the numbers that the scripts show." |

Pattern across all three procedural documents: contractions, em-dash joins
of independent clauses, passive voice for every completed action ("is
reviewed," "was chosen," "is superseded"), and code comments standing in
for instruction sentences instead of following one. None of this is a
judgment about the writing on its own terms — it reads fine as ordinary
technical English. It is not ASD-STE100.

## Part 2 — the conceptual text (out of scope by design)

Two real sentences, to show why bringing this content into compliance is
not a matter of trimming adjectives.

From `README.md` §1:

> Heat a magnet and it is ordered below `T_c` and disordered above; exactly
> at `T_c` — and only there — correlations become scale-free, fluctuations
> occur at every size, and the response to a small perturbation has no
> characteristic scale.

One sentence, three independent clauses joined by a semicolon and two
commas, 37 words against a 25-word ceiling for descriptive text, an
em-dash parenthetical, and four technical terms (`scale-free`, `measure-zero`
two sentences later, `perturbation`, `characteristic scale`) that sit
outside the ~875-word general dictionary. STE's allowance for technical
vocabulary covers naming a manufacturer's own parts and systems, consistently,
once defined — not importing a physics subfield's terminology.

From `PHILOSOPHY.md` §1:

> The sandpile is strongest exactly where a philosophical question can be
> reduced to a claim about aggregation — what collective pattern follows
> from a stated micro-rule — and weakest wherever the question needs
> semantic content (truth, meaning, justification) that the model has no
> representation for.

Same pattern at greater length (43 words against the same 25-word ceiling), plus abstract nouns with no STE
equivalent (`aggregation`, `semantic content`, and `epistemic`,
`verisimilitude`, `incommensurability` elsewhere in the same document). These
are not stylistic flourishes to be cut; they are the content. A
compliant rewrite would not simplify this paragraph, it would remove the
claim it makes.

**Conclusion for Part 2: do not convert these documents to STE.** The
standard was built on the premise that procedural ambiguity can injure
someone; that premise does not apply to an argument about Kitcher and
Zollman, and forcing the vocabulary constraint onto it would cost the
content for no corresponding safety benefit.

## Recommendation

1. Leave `README.md` §1–§6, `ANALYSIS.md`, `PHILOSOPHY.md`,
   `NOVELTY_RESULTS.md`, and the conceptual/philosophy vault notes as
   standard technical English. STE does not apply to them.
2. Bring the three procedural documents into actual compliance: one
   imperative instruction sentence per action, active voice throughout, no
   contractions, no em-dash sentence joins, approved vocabulary only
   (replace "tweak," "errors on," hedge adverbs), sentences at or under the
   20-word instruction limit.
3. Since the three procedural documents currently repeat each other with
   small wording differences, fixing one and linking the other two to it
   (rather than maintaining three slightly different copies) would remove
   the risk of them drifting out of agreement with each other, independent
   of the STE question.

This review did not change any file. Tell me if you want item 2 applied —
I would rewrite `obsidian-vault/Running the Code.md` to full compliance,
then point `README.md` §7 and `tutorial.html`'s run block at it instead of
each keeping their own copy.

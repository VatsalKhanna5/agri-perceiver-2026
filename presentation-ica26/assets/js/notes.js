/* Speaker notes. Kept apart from the slides on purpose: the deck stays clean,
   and the presenter view (presenter.html) reads this file.

   t      target seconds on this slide (they sum to 11:05, leaving ~55 s of the
          12:00 slot as buffer: live delivery always runs long)
   say    the spoken script, as paragraphs; **bold** = land this phrase
   cue    stage directions: clicks, pauses, where to point
   anchor the one sentence to fall back on if you blank                  */

window.NOTES = [
  {
    t: 25,
    anchor: "A small model that reads a leaf and writes the report.",
    cue: ["Stand still. Let the title sit for a beat before speaking."],
    say: [
      "Good morning. I'm Vatsal Khanna from NIT Jalandhar. This is joint work with Samridhi Saini, Davinder Singh, and Kumkum Kwatra of Annam.ai at IIT Ropar.",
      "Our paper is **AgriPerceiver**: a small vision–language model that looks at a single leaf and writes a **structured diagnostic report**. In the next twelve minutes I'll show why the report matters more than the label, how we made it cheap, what it gets right, and where it still falls short.",
    ],
  },
  {
    t: 50,
    anchor: "The bottleneck isn't the camera. It's the expert's time.",
    cue: ["Gesture to the leaf on the right.", "Slow down on the question at the bottom: it is the thesis of the talk."],
    say: [
      "Start with a picture. A farmer, say here in Punjab, holds a phone over a leaf with brown spots and yellow halos. **What should come back?**",
      "Plant pathogens and pests take **20 to 40 percent** of global crop yield every year. The person who can actually read this leaf is an agronomist, and there are **hundreds of millions** of smallholder farms. Everyone has a camera. The bottleneck is the **expert's time**.",
      "So our question was simple: can a **small** model read a leaf the way an agronomist writes a report?",
    ],
  },
  {
    t: 50,
    anchor: "A label can't be acted on. A report can, and it can be checked.",
    cue: ["Point at the JSON fields top to bottom as you name them.", "Touch the brackets on the right: classification, regression, language."],
    say: [
      "Most plant-disease work answers with a label: “Northern Leaf Blight.” Useful, but **a label is not a diagnosis**. It doesn't say how bad it is, what you saw, why you think so, or what to do this week.",
      "On the right is a real output from our model, from the paper's appendix: the diagnosis, the pathology type, a **severity of 0.62**, a confidence, the symptoms, the reasoning, and an action: apply propiconazole within 48 hours.",
      "It's really three problems at once: **classification, regression, and language**. We learn all three with a single decoding objective. A structured report is something a farmer can act on, and something an expert can **check and disagree with**.",
    ],
  },
  {
    t: 45,
    anchor: "Resolution, structure, scale.",
    cue: ["One card per sentence, left to right."],
    say: [
      "Why not just ask a big general-purpose VLM? We hit three walls.",
      "**One: lesions are tiny.** Resize a photo to 384 pixels and early water-soaking disappears. Keeping it takes more visual tokens, and attention cost grows with the **square** of the token count.",
      "**Two: chat models write prose.** Software needs a schema. As you'll see, prompting gets you valid JSON. It doesn't get you the right content.",
      "**Three: scale.** Seven to eight billion parameters. Fully fine-tuning them is out of reach for most agricultural labs, and it risks forgetting.",
    ],
  },
  {
    t: 35,
    anchor: "See sharp. Think small. Train little.",
    cue: ["Pause after each line. This slide is meant to be remembered, not read."],
    say: [
      "Our answer fits in three lines.",
      "**See sharp**: cut the image into tiles, so the model sees about twice the resolution.",
      "**Think small**: squeeze thousands of visual tokens into **128** learned latents.",
      "**Train little**: freeze both big backbones and train only about **nine percent** of the weights.",
      "Everything that follows is one of these three lines.",
    ],
  },
  {
    t: 70,
    anchor: "Tiles in, 128 latents across, JSON out. Grey frozen, green trained.",
    cue: ["Starts with only the leaf.", "→ CLICK 1: tiles + SigLIP", "→ CLICK 2: perception bridge", "→ CLICK 3: Phi-3 + JSON", "Then → to advance."],
    say: [
      "Here is the whole model. Green is trained; grey is frozen.",
      "[click] **See sharp.** The input, at any resolution, becomes five tiles: one global view and four quadrants, each 384 squared. A frozen SigLIP encoder turns each tile into 729 patch tokens.",
      "[click] **Think small.** That's 3,645 tokens, far too many to hand to a language model. So the **perception bridge**, the only large part we train, first adds a learned **tile embedding** so every patch remembers which quadrant it came from. Then it projects to the language model's width, and a two-block **Perceiver Resampler** lets 128 learned queries cross-attend to all 3,645 tokens. Out come **128 latents**.",
      "[click] **Speak in schema.** Those latents are spliced into the prompt of a frozen **Phi-3-mini**, 3.8 billion parameters, with small LoRA adapters, and it writes the seven-field JSON.",
    ],
  },
  {
    t: 45,
    anchor: "Quadratic versus linear: 27.5× at five tiles, 56× at ten.",
    cue: ["Left: the dots folding into the green grid.", "Right: trace the red curve up, then the green line along the floor."],
    say: [
      "Why the bottleneck matters. On the left, **3,645 tokens fold into 128**, 28.5 times fewer.",
      "On the right, attention cost per layer as you add tiles. Full self-attention is **quadratic**, the red curve. The Perceiver is **linear**, the green line hugging the floor. At our five tiles that's 163 GFLOPs against 5.9: **27.5 times cheaper**. At ten tiles the gap is **56 times**.",
      "And because the language model always sees exactly 128 latents, its cost doesn't change at all as resolution grows. To be precise, these are analytical attention FLOPs with projections excluded, not measured wall-clock.",
    ],
  },
  {
    t: 45,
    anchor: "A teacher labelled 117K images; we kept what it was sure of, and weighted by it.",
    cue: ["Follow the ribbons left to right.", "Point at the red drop-out: we threw away 14%."],
    say: [
      "Data. We had **117,635** leaf images and no expert-written reports, so we used a **teacher**. Gemma-3-12B, quantised to four bits, wrote a seven-field report for every image.",
      "We didn't trust its self-reported confidence. Instead we computed confidence from its own **token probabilities** and kept reports above 0.5. That's 86 percent: 96,239 to train on and **5,062 held out** for test. Each training sample is weighted by that confidence: easy, medium, hard.",
      "Training has two stages. First we align only the bridge. Then we add LoRA and specialise on the JSON. About a day in total on **one H200**, with both backbones frozen throughout.",
    ],
  },
  {
    t: 60,
    anchor: "Everyone writes valid JSON. Only one writes the right thing.",
    cue: ["Point at the greyed JSON / Schema columns first.", "Then the green row; read the four big numbers."],
    say: [
      "Results on the 5,062 held-out leaves, against three general VLMs (LLaVA-NeXT, InternVL2, Qwen2-VL), all given the **identical** structured prompt.",
      "First, look at the grey columns. **Every model writes valid JSON**, 99.7 to 100 percent. Structure is not the differentiator. The difference is **what the JSON says**.",
      "Composite **0.810** against 0.615. Pathology-type F1 **1.9 times** the best baseline. Severity error **3.3 times lower**, with a correlation of 0.86. And this with **9.2 percent** of the parameters trained, in a model smaller than every baseline.",
    ],
  },
  {
    t: 45,
    anchor: "Fine where it's easy, collapse where it matters. Hardest classes are rarest.",
    cue: ["Sweep left to right: two tall groups, then the cliff.", "Point at the n = labels under the axis."],
    say: [
      "Now pathology type, class by class. The generalists do fine where it's easy: **fungal and unknown**, the two largest and most visually distinctive classes.",
      "Then they fall off a cliff. On **bacterial and viral** leaves every baseline is at or below 0.025 F1. On pest, the best baseline reaches 0.23. AgriPerceiver is the **only model above 0.5 on all six**.",
      "Look at the counts underneath: the hardest classes are also the **rarest**, which is exactly where an agronomist's attention is most valuable.",
    ],
  },
  {
    t: 45,
    anchor: "Confusions of biology, not of format.",
    cue: ["Point to the diagonal, then to each red-outlined cell."],
    say: [
      "Now where we're still wrong. The diagonal is strong: 84 percent on fungal, 76 unknown, 71 deficiency. But bacterial and viral sit near **52 percent**.",
      "The red cells tell the story. **29 percent of viral leaves are called deficiency.** Diffuse yellowing, chlorosis, looks the same whether a virus or a missing nutrient caused it.",
      "And **21 to 23 percent** of bacterial and pest leaves are called fungal. At 384 pixels per tile, an early lesion looks like fungal necrosis.",
      "These are **confusions of biology, not of format**, and they tell us exactly what data to collect next.",
    ],
  },
  {
    t: 40,
    anchor: "Says 0.80, right 57% of the time. Fix: temperature scaling.",
    cue: ["Green is what happened, the black tick is what it said, red hatching is the gap.", "Point at the shaded LARGEST GAP bin: says 0.80, right 0.57."],
    say: [
      "A report is only actionable if its confidence is honest. Ours is **overconfident**: ECE 0.139. In the 0.7–0.8 bin, which adds the most to that error, the model says **0.80** and is right **57 percent** of the time. That's inherited, because the training targets carry the teacher's own confidence.",
      "For context, InternVL2 and Qwen2-VL sit near 0.38. LLaVA looks perfectly calibrated only because it answers “unknown” for almost everything. That's **not calibration**.",
      "The fix is standard and cheap, **temperature scaling**, and we haven't done it yet.",
    ],
  },
  {
    t: 50,
    anchor: "I'd rather you hear these from me.",
    cue: ["Slower pace. Eye contact. This slide earns trust."],
    say: [
      "Let me be direct about what this paper **does not** prove.",
      "**One:** our labels come from a model, not from agronomists. Every number you've seen is agreement with Gemma-3, and its biases are inside our ground truth.",
      "**Two:** our baselines were prompted, not fine-tuned. So we've shown that specialisation beats prompting, not that our architecture beats a fine-tuned generalist.",
      "**Three:** the spatial components barely move the aggregate. Removing AnyRes costs 0.004 composite, and removing tile embeddings costs nothing measurable. The ablations that need retraining aren't done.",
      "**Four** is overconfidence. **Five:** one photo, one moment, no field trial. I'd rather you hear these **from me**.",
    ],
  },
  {
    t: 35,
    anchor: "Calibrate and get expert labels first. Then harder classes, time, other crops.",
    cue: ["Trace the curve left to right with your hand."],
    say: [
      "So, the road from here, in order.",
      "**Next:** calibrate, and build an **agronomist-validated gold test set**. That's the single most important item.",
      "**Soon:** go after the chlorosis and early-lesion confusions with targeted data, and finish the ablations, including a fine-tuned baseline.",
      "**On the horizon:** following a leaf over time, transferring across crops, and eventually pilots in real fields, **beside agronomists, not instead of them**.",
    ],
  },
  {
    t: 25,
    anchor: "Every leaf is a report waiting to be read.",
    cue: ["Say the headline, pause, then thank.", "Stay on this slide for Q&A. Backup slide (→) has parameters, hyperparameters, composite weights."],
    say: [
      "To close: **every leaf is a report waiting to be read.** See sharp, think small, train little, and you can read it on a single GPU.",
      "Thank you to our mentors at NIT Jalandhar, and to Annam.ai at IIT Ropar for the compute and the domain expertise. I'm happy to take questions.",
    ],
  },
  {
    t: 0,
    backup: true,
    anchor: "Backup: open only if asked.",
    cue: ["Use for: parameter counts, hyperparameters, composite weighting."],
    say: [
      "Trained parameters: the bridge, 391 million, plus LoRA, 35 million. That's 426 million of 4.65 billion, 9.2 percent. SigLIP and Phi-3 never receive gradients.",
      "Composite weights put the most on pathology type (0.20). The three BERTScore terms together get 0.35. Doubling any one family of weights still leaves a margin of at least +0.161 over the best baseline.",
    ],
  },
];

/* Likely questions, with short honest answers. Three minutes of Q&A means
   two or three questions: answer in two sentences, then stop. */
window.QA = [
  {
    q: "Your ground truth comes from Gemma-3. Aren't you just distilling the teacher?",
    a: "Partly, yes, and that's by design: we distil a 12B teacher into a model that trains 9.2% of its weights and runs on one GPU. But it means our accuracy numbers measure agreement with the teacher, not with agronomists. That's limitation one, and the expert gold set is our first next step.",
  },
  {
    q: "Why not fine-tune the baselines on your data?",
    a: "Compute. Fully fine-tuning a 7–8B model was beyond our budget. The comparison we report is the one most groups actually face, a domain-trained small model against prompted large ones. A fine-tuned generalist is the comparison we still owe.",
  },
  {
    q: "Removing AnyRes changes the composite by only 0.004. Why keep it?",
    a: "That's a runtime ablation: the model was trained with tiles and then shown five copies of the global view, without retraining. It moves F1, fuzzy-match and MAE in the same direction, but the effect is small, and we say so. A properly retrained ablation is on the list.",
  },
  {
    q: "Baselines get ~0 diagnosis fuzzy-match. Isn't that just naming mismatch?",
    a: "Largely, yes. Token-Jaccard against the teacher's disease names rewards matching its vocabulary, which our model learned. That's why I'd point to pathology-type F1, a fixed six-class label, where the gap is still 1.9×.",
  },
  {
    q: "The paper says baselines get pest F1 ≤ 0.017, but the figure shows InternVL2 near 0.23.",
    a: "You're right, thank you. That bound holds for Qwen2-VL and LLaVA. InternVL2 reaches 0.227 on pest, as the figure shows. The ≤ 0.025 claim holds strictly for bacterial and viral.",
  },
  {
    q: "Why Phi-3-mini and SigLIP?",
    a: "Phi-3-mini's 128K context fits long JSON without truncation, and at 3.8B the whole inference stack fits a single 24 GB GPU. SigLIP-SO400M is a strong frozen encoder at 384 px.",
  },
  {
    q: "Why 128 latents? Did you try other counts?",
    a: "128 follows Flamingo-style resamplers and keeps the language model's input fixed and small. 64 and 256 need retraining, so they're planned but not run.",
  },
  {
    q: "Does it run on a phone? What's the latency?",
    a: "Not yet measured. The FLOPs on the slide are analytical. The full stack fits one 24 GB GPU. On-device deployment, with quantisation, is future work.",
  },
  {
    q: "What does the data look like? Lab images or field images?",
    a: "117,635 images in 57 categories, about 14 crop species, drawn from PlantVillage, Kaggle and other open repositories, so many have controlled backgrounds. Generalisation to messy field photos is untested, which is why field pilots are on the roadmap.",
  },
  {
    q: "Isn't the composite score arbitrary?",
    a: "The weights are a choice, so we stress-tested them. Doubling the structural, classification or semantic weights in turn keeps us ahead by at least +0.161. The individual metrics are all on the results slide too.",
  },
  {
    q: "Where does severity ground truth come from?",
    a: "Also from the teacher's report, a value in [0, 1]. So MAE 0.067 and r 0.855 measure agreement with the teacher's severity judgement. Expert severity scoring is part of the gold set.",
  },
];

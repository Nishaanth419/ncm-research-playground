// Static content for the Learn page. Numbers are this project's measured results
// (docs/EXPERIMENTS.md); paper facts are from the paper and the official README.

export const PAPER = {
  title: 'Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation',
  authors: 'Hermann Kumbong, Ishan S. Khare, Iddah Mlauzi, Andrew Woen, Charles Ding, E. Kelly Buchanan, Simran Arora, Azalia Mirhoseini, Christopher Ré',
  year: 2026,
  links: [
    { label: 'Paper (arXiv)', href: 'https://arxiv.org/abs/2610.14162' },
    { label: 'Official code', href: 'https://github.com/HazyResearch/ncm' },
    { label: 'Models & data', href: 'https://huggingface.co/collections/hazyresearch/ncm' },
    { label: 'Blog post', href: 'https://hazyresearch.stanford.edu/blog/2026-10-12-ncm' },
  ],
  tasks: ['TinyStories', 'OpenWebText', 'LM1B', 'XSum summarisation', 'WMT14 De→En translation', 'Sudoku'],
}

export interface KeyIdea {
  title: string
  body: string
}

export const KEY_IDEAS: KeyIdea[] = [
  {
    title: 'Text becomes levels of codes',
    body: 'A tokenizer network (a multiscale residual VQ-VAE) compresses a passage into a stack of levels. Each level is a short list of discrete codes picked from a codebook; together they describe the whole passage. These codes are the "concepts".',
  },
  {
    title: 'Coarse to fine',
    body: 'The top level is a single code for the whole TinyStories passage, the next has 8, then 16, and so on up to 128. Early levels carry the gist (who, what kind of story); later levels add the exact words.',
  },
  {
    title: 'A whole level at once',
    body: 'A transformer predicts the next level from the levels above it, and every code of a level comes out of one forward pass, in parallel. A 128-token story takes 11 passes instead of up to 128.',
  },
  {
    title: 'Borrowed from images',
    body: 'The idea follows next-scale prediction for images (VAR), where a picture is generated from a blurry thumbnail up to full resolution. NCM applies the same coarse-to-fine recipe to text.',
  },
  {
    title: 'The trade-offs',
    body: 'Fewer sequential steps make it fast, especially in batches. But nothing can be shown until the last level is done (no word-by-word streaming), and in our test the text was less fluent than a word-by-word model of the same domain.',
  },
]

export interface Finding {
  label: string
  value: string
  hint: string
}

export const FINDINGS: Finding[] = [
  { label: 'Paper result reproduced', value: '16.23 ± 0.48', hint: 'gen-PPL on TinyStories, 5 seeds × 256 stories, laptop CPU. The paper reports 15.41.' },
  { label: 'Faster end to end', value: '1.42× / 1.79×', hint: 'Median time for a full story vs. TinyStories-33M, at batch 1 / batch 4 (307 ms vs. 437 ms at batch 1).' },
  { label: 'Sequential passes', value: '11 vs. 128', hint: 'NCM fills a 128-token canvas in 11 transformer passes; a word-by-word model needs up to 128.' },
  { label: 'Fluency (lower is better)', value: '13.48 vs. 9.74', hint: 'gen-PPL under GPT-2-large, NCM vs. TinyStories-33M, one 4-sample run. The word-by-word model wins.' },
  { label: 'First output', value: '307 ms vs. 4 ms', hint: 'NCM shows the whole story at once; the word-by-word model streams its first token almost immediately.' },
  { label: 'Faithful to the official code', value: 'identical', hint: 'This app reproduces the official script outputs exactly on TinyStories, XSum and WMT14 test items.' },
]

export const CAVEATS = [
  'The paper\'s 9–31× speedups were measured on an H100 GPU against diffusion language models. Ours are laptop-CPU numbers against a word-by-word model: a different experiment, not a contradiction.',
  'The two story models differ in size (125M vs. 68.5M parameters), tokenizer and training run.',
  'The quality comparison is a single small run; the speed numbers are medians of 10 runs.',
]

export interface Video {
  id: string
  title: string
  channel: string
  note: string
}

// Verified to exist and be embeddable via YouTube oEmbed (2026-11-04). Third-party explainers, not reviewed for accuracy.
export const VIDEOS: Video[] = [
  { id: 'z3CoFbZhYGA', title: 'NCM: Fast Parallel Coarse-to-Fine Text Generation', channel: 'AI Research Roundup', note: 'Walk-through of the paper' },
  { id: 'cysJAICGEiA', title: "Stanford's NCM (Next-Concept Modeling): Generate Text Without Predicting Every Token?", channel: 'AgenticEngineering', note: 'The hierarchy of codes, explained' },
  { id: 'J0tqdxQ8PHg', title: 'Next-Concept Modeling Explained: Can AI Move Beyond Tokens?', channel: 'KeepItTechie', note: 'Beginner-friendly overview' },
  { id: 'SMYzc9FMXj0', title: 'NCM Just Killed LLM: This Time, It Might Be for Real', channel: 'Cloud X Berry', note: 'How it works and what it means for developers' },
  { id: 'vLqswKyHS64', title: 'NCM Is this the End of LLMs?', channel: 'Kai', note: 'A skeptical look at the "31× faster" claim' },
  { id: 'yQvELPjmyn0', title: 'Vector-Quantized Variational Autoencoders (VQ-VAEs)', channel: 'DeepBean', note: 'Background: the kind of tokenizer NCM builds on' },
]

export const GLOSSARY: { term: string; def: string }[] = [
  { term: 'Token', def: 'A piece of text a model works with: a word or part of a word.' },
  { term: 'Autoregressive model', def: 'A traditional language model (like GPT). It writes left to right, one token per step, each step depending on all the previous ones.' },
  { term: 'Forward pass', def: 'One run of the neural network. The number of sequential passes is what limits generation speed.' },
  { term: 'VQ-VAE', def: 'A network that compresses data into discrete codes from a learned codebook and can decode the codes back. NCM uses a multiscale, residual version for text.' },
  { term: 'Codebook', def: 'The fixed list of vectors a level can choose from (4,096 per level for TinyStories). A code is an index into it.' },
  { term: 'Level', def: 'One row of the NCM hierarchy. Each level refines what the levels above it described.' },
  { term: 'Diffusion language model', def: 'Generates text by starting from noise or masks and refining all positions over many steps. The paper\'s main speed comparison.' },
  { term: 'gen-PPL', def: 'Generative perplexity: how surprised a large judge model (GPT-2-large) is by the generated text. Lower means more fluent.' },
  { term: 'Entropy', def: 'How varied the words are. Too low means repetitive text; it guards against gaming gen-PPL.' },
]

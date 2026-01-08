# Day 5: NLP & Text Processing Fundamentals

## Overview

Natural Language Processing (NLP) has undergone a revolution similar to computer vision. Understanding NLP is critical for Principal AI Architects because:
- Most enterprise data is text (emails, documents, logs, customer feedback)
- LLMs (Large Language Models) now dominate AI discussions
- NLP techniques apply beyond text (time-series, code, structured data)
- Interview questions heavily feature NLP system design

**The NLP Revolution:**
- Pre-2018: Feature engineering + traditional ML
- 2018: BERT (bidirectional transformers)
- 2020: GPT-3 (massive scale language models)
- 2022: ChatGPT (conversational AI mainstream)
- 2023+: LLMs as general-purpose AI

---

## 1. Why NLP is Different from Other ML

### Unique Challenges

**1. Discrete, Symbolic Data**
```
Images: Continuous pixel values (0-255)
Text: Discrete tokens (words from vocabulary)

Problem: How do we represent words as numbers?
- Can't use raw encoding: "cat" = 1, "dog" = 2 implies dog > cat
- Need meaningful representations
```

**2. Sequential Nature**
```
Order matters:
- "Dog bites man" ≠ "Man bites dog"
- Unlike images where spatial location flexible

Need models that understand sequence:
- RNNs, LSTMs (older)
- Transformers (modern)
```

**3. Variable Length**
```
Sentences have different lengths:
- "Hello" (1 word)
- "The quick brown fox jumps over the lazy dog" (9 words)

Problem: Neural networks expect fixed input size
Solutions: Padding, packing, attention mechanisms
```

**4. Context and Ambiguity**
```
Same word, different meanings:
- "I went to the bank" (financial institution)
- "I sat by the river bank" (land beside water)

Context determines meaning
Need contextual representations (not static embeddings)
```

**5. Long-Range Dependencies**
```
"The cat that scared the dog that chased the mouse ran away"

"The cat" is subject of "ran away" (7 words apart)
Need to capture long-range relationships
Traditional RNNs struggle, Transformers excel
```

---

## 2. Text Preprocessing Pipeline

### Why Preprocess?

Raw text is messy:
- Inconsistent capitalization, punctuation
- Special characters, URLs, emails
- Multiple languages, encoding issues
- Spelling errors, slang, abbreviations

Goal: Clean, normalize text for consistent model input

### Preprocessing Steps

**Step 1: Text Cleaning**

**Lowercasing:**
```
Original: "The Cat sat on the MAT"
Lowercased: "the cat sat on the mat"

Why:
- Reduces vocabulary size (Cat, cat, CAT → cat)
- Improves generalization

When NOT to:
- Named Entity Recognition (John vs john)
- Sentiment analysis (WOW vs wow)
```

**Remove Special Characters:**
```
Original: "Hello!!! How are you? 😊 #NLP"
Cleaned: "Hello How are you NLP"

Remove: Punctuation, emojis, hashtags, URLs, emails

Trade-off:
- Removes noise
- But loses information (punctuation affects meaning)
- Modern models keep punctuation (context matters)
```

**Handling Numbers:**
```
Options:
1. Remove: "I have 3 cats" → "I have cats"
2. Normalize: "I have 3 cats" → "I have <NUM> cats"
3. Keep: "I have 3 cats" (modern approach)

Choice depends on task:
- Remove: Document classification (numbers not relevant)
- Normalize: General NLP (preserve that number exists)
- Keep: Question answering, math problems
```

**Step 2: Tokenization**

**Definition:** Split text into units (words, subwords, characters)

**Word Tokenization:**
```
Sentence: "Hello, world! How are you?"
Tokens: ["Hello", ",", "world", "!", "How", "are", "you", "?"]

Methods:
- Split by spaces: Simple but wrong for punctuation
- Rule-based: Handle apostrophes, hyphens (I'm → I'm or I + am?)
- TreeBank tokenizer: Standard for English

Challenges:
- Contractions: "don't" → "do" + "n't" or keep as "don't"?
- Hyphenated: "state-of-the-art" → one token or multiple?
- Punctuation: Keep or remove?
```

**Subword Tokenization (Modern Approach):**
```
Why:
- Vocabulary size limited (typically 30K-50K tokens)
- Out-of-vocabulary (OOV) words problematic
- Morphology: "unhappiness" = "un" + "happy" + "ness"

Algorithms:
1. Byte-Pair Encoding (BPE) - GPT uses this
2. WordPiece - BERT uses this
3. SentencePiece - Language-agnostic

Example (BPE):
Word: "unhappiness"
Subwords: ["un", "happi", "ness"]

Word: "antiestablishmentarianism"
Subwords: ["anti", "establish", "ment", "arian", "ism"]

Benefits:
- Fixed vocabulary handles any word
- Captures morphology
- Shares representations (happy, happiness, unhappy)
```

**Character Tokenization:**
```
Sentence: "Hello"
Tokens: ["H", "e", "l", "l", "o"]

Pros:
- Tiny vocabulary (26 letters + punctuation)
- No OOV problem
- Handles misspellings naturally

Cons:
- Very long sequences (10× longer than word-level)
- Harder to learn (must learn what constitutes words)
- Computationally expensive

Use case: Spell checking, OCR correction
```

**Step 3: Stop Word Removal (Optional)**

```
Stop words: Common words with little meaning
Examples: "the", "a", "an", "is", "are", "was", "were"

Original: "the cat is on the mat"
After removal: "cat mat"

When to remove:
- Traditional ML (bag-of-words, TF-IDF)
- Reduces dimensionality
- Speeds up processing

When NOT to remove:
- Deep learning models (context matters)
- "to be or not to be" → "be not be" (loses meaning!)
- Modern approach: Keep all words, let model learn
```

**Step 4: Stemming vs Lemmatization**

**Stemming:** Crude heuristic to remove suffixes
```
running → run
flies → fli (incorrect!)
better → better (doesn't change)

Algorithm: Porter Stemmer (rule-based)

Pros: Fast, simple
Cons: Often incorrect, not real words
```

**Lemmatization:** Convert to dictionary form (lemma)
```
running → run
flies → fly
better → good (if adjective context)

Requires: POS (Part-of-Speech) tagging, dictionary lookup

Pros: Linguistically correct, real words
Cons: Slower, needs POS tagger

Modern approach: Skip both, use subword tokenization
```

**Step 5: Handling Special Cases**

**URLs and Emails:**
```
"Visit https://example.com or email me@example.com"

Options:
- Remove: "Visit or email"
- Normalize: "Visit <URL> or email <EMAIL>"
- Keep: (if relevant to task)
```

**Hashtags and Mentions (Social Media):**
```
"#MachineLearning is awesome! @OpenAI"

Options:
- Remove symbols: "MachineLearning is awesome OpenAI"
- Keep: "#MachineLearning is awesome! @OpenAI"
- Split hashtags: "Machine Learning is awesome OpenAI"
```

**Multiple Languages:**
```
"I love NLP! C'est magnifique! 很好!"

Challenges:
- Different scripts (Latin, Chinese, Arabic)
- Different tokenization rules
- Mixed language in same document

Solutions:
- Language detection (langdetect library)
- Language-specific tokenizers
- Multilingual models (mBERT, XLM-R)
```

### Modern Preprocessing (Minimal)

**Traditional NLP (pre-2018):**
```
Extensive preprocessing required:
1. Lowercase
2. Remove punctuation
3. Remove stop words
4. Stem/lemmatize
5. Handle special cases

Why: Feature engineering critical (bag-of-words, TF-IDF)
```

**Modern NLP (Transformers):**
```
Minimal preprocessing:
1. Text cleaning (remove HTML, fix encoding)
2. Tokenization (subword tokenization)
3. That's it!

Why: Models learn from raw text, context matters
- BERT sees punctuation (affects meaning)
- GPT sees capitalization (names important)
- Models handle rare words (subword tokenization)
```

**Best Practice:**
For modern transformers: Do as little preprocessing as possible. Let the model learn!

---

## 3. Text Representation - From One-Hot to Embeddings

### 3.1 One-Hot Encoding (Naive Approach)

```
Vocabulary: ["cat", "dog", "mouse"]

"cat":   [1, 0, 0]
"dog":   [0, 1, 0]
"mouse": [0, 0, 1]

Problems:
1. Vocabulary explosion: 100K words → 100K-dimensional vectors
2. No similarity: Distance("cat", "dog") = Distance("cat", "rock")
3. Sparse: Mostly zeros, inefficient
4. No generalization: "kitten" is OOV (never seen)

Used only for: Small vocabularies (POS tags, NER labels)
```

### 3.2 Bag of Words (BoW)

```
Sentence: "the cat sat on the mat"
Vocabulary: ["cat", "dog", "mat", "sat", "the", "on"]

Vector: [1, 0, 1, 1, 2, 1]
        cat dog mat sat the on

Count occurrences of each word

Problems:
1. Loses order: "cat sat mat" = "mat sat cat"
2. Ignores context
3. Still sparse, high-dimensional
4. Common words dominate (the, a, is)

When to use: Traditional ML baselines, simple classification
```

### 3.3 TF-IDF (Term Frequency-Inverse Document Frequency)

```
Idea: Weight words by importance
- TF: How often word appears in document (term frequency)
- IDF: How rare word is across corpus (inverse document frequency)

TF-IDF = TF × IDF

Example:
Word "the": High TF, Low IDF (appears everywhere) → Low score
Word "quantum": Low TF, High IDF (rare, specific) → High score

Formula:
TF(word, doc) = count(word in doc) / total words in doc
IDF(word) = log(total documents / documents containing word)
TF-IDF(word, doc) = TF × IDF

Result: Downweights common words, upweights rare/informative words

Problems:
1. Still sparse
2. No semantic similarity (car, automobile unrelated)
3. Loses order and context

When to use: 
- Document search/retrieval
- Traditional ML baseline
- Fast, interpretable
```

### 3.4 Word Embeddings (The Breakthrough)

**Key Idea:** Represent words as dense vectors in continuous space

```
"cat":   [0.2, -0.4, 0.7, 0.1, ...]  (300 dimensions)
"dog":   [0.3, -0.3, 0.6, 0.2, ...]
"car":   [-0.5, 0.8, -0.2, 0.9, ...]

Properties:
- Dense (no zeros)
- Low-dimensional (50-300 vs 100K)
- Semantic similarity: Similar words → Similar vectors
- Arithmetic: king - man + woman ≈ queen
```

**How Embeddings Capture Meaning:**
```
Distributional Hypothesis: "A word is characterized by the company it keeps"

Words appearing in similar contexts have similar meanings:
- "The cat sat on the mat"
- "The dog sat on the mat"

"cat" and "dog" appear in similar contexts → Similar embeddings
```

---

## 4. Word2Vec - Learning Word Embeddings

### The Innovation (2013)

**Traditional:** Words are discrete symbols, no relationship
**Word2Vec:** Learn word representations by predicting context

### Two Architectures

**Skip-Gram:** Given word, predict surrounding context words
```
Sentence: "the quick brown fox jumps"
Window size: 2

Training examples:
Input: "brown" → Predict: ["the", "quick", "fox", "jumps"]

Learns: Words with similar contexts get similar embeddings
```

**CBOW (Continuous Bag of Words):** Given context, predict center word
```
Input: ["the", "quick", "fox", "jumps"] → Predict: "brown"

Faster to train than Skip-Gram
Skip-Gram works better for rare words
```

### How Training Works (Simplified)

```
1. Initialize random embeddings (300-dim vectors)
2. For each word in corpus:
   - Get its context window
   - Predict context words from center word (Skip-Gram)
   - Compare prediction to actual context
   - Update embeddings to improve prediction
3. After training on billions of words:
   - Similar words have similar embeddings
   - Semantic relationships emerge
```

**Negative Sampling (Efficiency Trick):**
```
Problem: Predicting from vocabulary of 100K words (softmax) is slow

Solution: Instead of predicting all words, use binary classification
- Positive examples: Actual context words
- Negative examples: Random words (not in context)
- Much faster: 5-20 negative samples vs 100K classes

Example:
Given "brown", positive: "fox", negatives: ["table", "Jupiter", "purple"]
Task: Is "fox" in context of "brown"? Yes (1)
      Is "table" in context of "brown"? No (0)
```

### Word2Vec Properties

**Semantic Similarity:**
```
Cosine similarity measures vector closeness:
similarity("cat", "dog") = 0.8 (high, both animals)
similarity("cat", "car") = 0.1 (low, unrelated)

Applications:
- Find similar words
- Document similarity
- Recommendation systems
```

**Analogies (Vector Arithmetic):**
```
king - man + woman ≈ queen
Paris - France + Italy ≈ Rome
walking - walk + swim ≈ swimming

Why this works:
- "king" - "man" = vector for "royalty + male"
- Add "woman" = "royalty + female"
- Closest word: "queen"

Not perfect but surprisingly effective!
```

**Limitations:**

```
1. Fixed representations:
   "bank" always same vector (can't distinguish financial vs river)
   
2. No handling of OOV words:
   "unhappiness" not in vocabulary → can't get embedding
   
3. Trained separately from task:
   Embeddings may not be optimal for specific task
   
4. No sentence-level representation:
   How to combine word embeddings into sentence embedding?
   Simple average loses information
```

---

## 5. FastText - Subword Embeddings

### The Problem Word2Vec Doesn't Solve

```
Out-of-vocabulary words:
Training: "happy", "happiness", "unhappy"
Test: "unhappiness" → No embedding!

Morphology ignored:
"teach", "teacher", "teaching" → Separate, unrelated embeddings
But they share meaning (all related to teaching)
```

### FastText Solution (2016)

**Key Idea:** Represent words as bag of character n-grams

```
Word: "where"
Character n-grams (n=3):
  - <wh
  - whe
  - her
  - ere
  - re>
  (< and > mark word boundaries)

Word embedding = Sum of n-gram embeddings

For OOV word "wherever":
  - Break into n-grams: <wh, whe, her, ere, rev, eve, ver, re>
  - Sum their embeddings
  - Get approximate embedding even though word never seen!
```

**Benefits:**

```
1. Handles OOV words:
   "unhappiness" → <un, unh, nha, hap, app, ppi, pin, ine, nes, ess, ss>
   Even if never seen, can construct embedding

2. Captures morphology:
   "teach", "teacher", "teaching" share n-grams (tea, eac, ach)
   → More similar embeddings

3. Handles misspellings:
   "happppy" → Similar n-grams to "happy"
   → Robust to typos

4. Works for rare words:
   Even with 1-2 examples, can learn from n-grams
```

**When to Use:**

- Morphologically rich languages (German, Turkish, Finnish)
- Social media text (lots of misspellings, slang)
- Domain-specific vocabulary (medical, legal) with OOV words
- Small training datasets (better generalization)

---

## 6. From RNNs to Transformers

### 6.1 Recurrent Neural Networks (RNNs)

**Idea:** Process sequences one word at a time, maintain hidden state

```
Sentence: "The cat sat"

Step 1: Process "The"
  Input: embedding("The")
  Hidden state: h1 (summarizes "The")

Step 2: Process "cat"
  Input: embedding("cat") + h1 (previous hidden state)
  Hidden state: h2 (summarizes "The cat")

Step 3: Process "sat"
  Input: embedding("sat") + h2
  Hidden state: h3 (summarizes "The cat sat")

Final h3 contains information about entire sentence
```

**Problems:**

```
1. Sequential processing (can't parallelize):
   Must process word 1 before word 2
   Training very slow

2. Vanishing gradients:
   Long sentences: gradient diminishes by time step 50
   Model forgets beginning of sentence

3. Short-term memory:
   Struggles with long-range dependencies
   "The cat that scared the dog ... ran away"
   By "ran away", forgot "cat" is subject
```

### 6.2 LSTMs (Long Short-Term Memory)

**Improvement over RNNs:** Gates control information flow

```
Three gates:
1. Forget gate: What to forget from cell state
2. Input gate: What new info to add
3. Output gate: What to output

Cell state: Separate from hidden state, stores long-term info

Result: Better at long-range dependencies than vanilla RNN
```

**Still Has Problems:**
```
1. Still sequential (slow training)
2. Still has limits on memory (not infinite)
3. Still struggles with very long documents
```

### 6.3 Attention Mechanism (The Key Innovation)

**Problem:**
```
Machine translation: "I love cats" → "J'aime les chats"

When translating "chats" (cats), model needs to look back at "cats"
RNN only has single hidden state (lossy summary)
Hard to keep all information from source sentence
```

**Solution: Attention**
```
Instead of single context vector, compute weighted average of ALL encoder states

When decoding "chats":
- Look at all English words
- Attend strongly to "cats" (high weight)
- Attend weakly to "I", "love" (low weights)

Attention weights: Model learns where to focus
```

**Attention Mechanism:**
```
Query: What we're trying to decode (current decoder state)
Keys: All encoder states (each word in input)
Values: All encoder states

Attention weight: How relevant each key is to query
Output: Weighted sum of values (based on attention weights)

Formula:
  Attention(Q, K, V) = softmax(Q · K^T / √d) · V
  
  Q · K^T: Dot product (similarity between query and each key)
  √d: Scaling factor (for stability)
  softmax: Convert to probabilities (weights sum to 1)
  · V: Weighted sum of values
```

**Why Attention is Powerful:**
```
1. Direct connections: Every output position can look at every input position
   - No vanishing gradient (short paths)
   - Perfect for long-range dependencies

2. Interpretable: Attention weights show which words model focuses on
   - Useful for debugging
   - Build trust (see what model attends to)

3. Parallelizable: All attention computations independent
   - Much faster than RNN sequential processing
```

---

## 7. Transformers - The Revolution (2017)

### "Attention Is All You Need"

**Key Insight:** Remove recurrence entirely, use only attention

**Architecture Overview:**
```
Input: "The cat sat on the mat"

1. Word embeddings + positional encoding
   (Positional encoding: Since no recurrence, need to inject position info)

2. Encoder (stack of 6-12 layers):
   Each layer:
   - Multi-head self-attention (words attend to each other)
   - Feed-forward network (position-wise)
   - Residual connections + LayerNorm

3. Decoder (for translation/generation):
   Each layer:
   - Masked self-attention (can't see future words)
   - Encoder-decoder attention (attend to source)
   - Feed-forward network

4. Output: Probabilities for next word
```

### Key Components

**1. Self-Attention**
```
All words attend to all words (including themselves)

"The cat sat on the mat"

When processing "sat":
- Attends to "cat" (who sat?)
- Attends to "on" (sat where?)
- Attends to "mat" (sat on what?)
- Attends to "the" (less relevant)

Every word gets context-aware representation
```

**2. Multi-Head Attention**
```
Why multiple heads?

Single attention: One way of relating words
Multiple heads: Multiple ways simultaneously

Head 1: Syntactic relationships (subject-verb)
Head 2: Semantic relationships (cat-animal)
Head 3: Positional relationships (nearby words)
... 
Head 8-12: Different aspects

Typical: 8-16 attention heads

Concatenate all heads → richer representation
```

**3. Positional Encoding**
```
Problem: Attention has no notion of order
"cat sat mat" = "mat sat cat" (without position info)

Solution: Add positional encodings to embeddings

PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

pos: Position in sequence
i: Dimension
d: Embedding dimension

Result: Each position gets unique encoding
Model learns to use position information
```

**4. Feed-Forward Networks**
```
After attention, each position processed independently:

FFN(x) = ReLU(x·W1 + b1)·W2 + b2

Two linear transformations with ReLU
Applied to each position separately
Adds non-linearity and capacity
```

**5. Residual Connections & Layer Normalization**
```
Residual: Output = LayerNorm(x + Sublayer(x))

Why:
- Enables training deep networks (like ResNet)
- Gradient flow (skip connections)
- Stable training

Layer Normalization:
- Normalizes across features (not batch)
- Stabilizes training
- Faster convergence
```

### Why Transformers Dominate

**1. Parallelizable:**
```
RNN: Process word 1, then word 2, then word 3 (sequential)
Transformer: Process all words simultaneously (parallel)

Result: 10-100× faster training
Enables training on massive datasets
```

**2. Long-Range Dependencies:**
```
RNN: Information flows through many steps (degrades)
Transformer: Direct connections (constant path length)

Result: Handles documents, not just sentences
No vanishing gradients
```

**3. Interpretable:**
```
Attention weights show what model attends to
Useful for debugging and building trust
```

**4. Scalable:**
```
Architecture scales beautifully:
- Bigger models (more layers, more heads)
- More data (billions of words)
- More compute (thousands of GPUs)

Result: GPT-3 (175B parameters), GPT-4 (1T+ parameters)
Performance keeps improving with scale
```

---

## 8. BERT - Bidirectional Encoder Representations (2018)

### The Pre-training Revolution

**Key Idea:** Pre-train on massive unlabeled data, fine-tune on specific tasks

**Traditional Approach:**
```
Task 1: Sentiment analysis → Train model from scratch
Task 2: Question answering → Train model from scratch
Task 3: NER → Train model from scratch

Problem: Each task needs lots of labeled data, slow
```

**BERT Approach:**
```
Step 1: Pre-train on unlabeled data (billions of words)
  - Learn general language understanding
  - Only needs to be done once

Step 2: Fine-tune on task-specific labeled data
  - Fast (hours instead of days)
  - Needs less labeled data (few thousand examples)

Works for: Sentiment, QA, NER, text classification, etc.
```

### BERT Pre-training Tasks

**Masked Language Model (MLM):**
```
Original: "The cat sat on the mat"
Masked: "The [MASK] sat on the [MASK]"
Task: Predict masked words

Forces model to understand context:
- "sat on the [MASK]" → probably "mat", "floor", "chair"
- Not "car", "sky" (doesn't make sense)

Why bidirectional?
- Model sees both left and right context
- "The [MASK] sat" (left context) + "sat on the" (right context)
- Better than left-to-right (GPT) for understanding tasks
```

**Next Sentence Prediction (NSP):**
```
Given two sentences:
Sentence A: "I love cats"
Sentence B: "They are cute" (IsNext) or "Paris is a city" (NotNext)

Task: Predict if B follows A

Why?
- Teaches sentence relationships
- Useful for QA, inference tasks
- Later replaced by better objectives (RoBERTa removed NSP)
```

### BERT Architecture

```
Base model:
- 12 Transformer encoder layers
- 768 hidden dimensions
- 12 attention heads
- 110M parameters

Large model:
- 24 layers
- 1024 hidden dimensions
- 16 attention heads
- 340M parameters

Input: [CLS] Sentence A [SEP] Sentence B [SEP]
  [CLS]: Classification token (represents entire input)
  [SEP]: Separator between sentences

Output: Contextual embeddings for each token
```

### Fine-Tuning BERT

**Task 1: Text Classification (e.g., Sentiment Analysis)**
```
Pre-trained BERT
  ↓
Add classification head (single linear layer)
  ↓
Train on labeled data (1K-10K examples)
  ↓
Use [CLS] token embedding for classification

Few epochs (2-4) sufficient
90%+ accuracy typical
```

**Task 2: Named Entity Recognition (Token Classification)**
```
Pre-trained BERT
  ↓
Add token classification head (linear layer per token)
  ↓
Train on labeled data
  ↓
Each token gets label (PERSON, ORG, LOC, O)

Example:
"John works at Google in Paris"
[PERSON] [O] [O] [ORG] [O] [LOC]
```

**Task 3: Question Answering**
```
Input: Question + Context paragraph
Output: Span in paragraph (start, end positions)

Pre-trained BERT
  ↓
Add two linear layers (start position, end position)
  ↓
Train on SQuAD dataset
  ↓
Extract answer span

Example:
Question: "Who founded Apple?"
Context: "Steve Jobs and Steve Wozniak founded Apple in 1976"
Answer: Span [0, 29] = "Steve Jobs and Steve Wozniak"
```

### BERT Impact

**State-of-the-Art Improvements:**
- SQuAD (QA): Human parity (was 10 points behind)
- GLUE (language understanding): 7% absolute improvement
- Most NLP tasks: Significant gains

**New Paradigm:**
Pre-train + fine-tune became standard
- Transfer learning for NLP (like ImageNet for CV)
- Democratized NLP (don't need massive compute for each task)

---

## 9. GPT - Generative Pre-trained Transformer

### GPT vs BERT: Key Differences

**BERT:** Encoder-only, bidirectional
```
Use case: Understanding tasks
- Classification, NER, QA
- Sees full context (left and right)
- Cannot generate text naturally
```

**GPT:** Decoder-only, unidirectional (left-to-right)
```
Use case: Generation tasks
- Text completion, dialogue, translation
- Sees only left context (autoregressive)
- Natural for generation
```

### GPT Pre-training: Language Modeling

**Task:** Predict next word given previous words
```
Input: "The cat sat on the"
Target: "mat"

Input: "The cat sat on the mat in"
Target: "the"

Train on billions of words
Learn to predict what comes next
```

**Why This Works:**
```
To predict next word, model must understand:
- Grammar (subject-verb agreement)
- Semantics (what makes sense)
- World knowledge (common associations)
- Context (maintain coherence)

Emergent capabilities: By learning to predict next word, model learns language!
```

### GPT Evolution

**GPT-1 (2018):**
```
Parameters: 117M
Pre-training: BooksCorpus (800M words)
Fine-tuning: Required for each task
Performance: Competitive with BERT
```

**GPT-2 (2019):**
```
Parameters: 1.5B (13× larger)
Pre-training: WebText (8M web pages)
Zero-shot: No fine-tuning needed!
Prompting: "Translate to French: I love cats →"

Controversy: "Too dangerous to release"
Actually released: Showed responsible AI practices
```

**GPT-3 (2020):**
```
Parameters: 175B (100× larger than GPT-2)
Pre-training: 570GB of text
Few-shot learning: Works with just examples in prompt
In-context learning: Learns from prompt (no parameter updates)

Capabilities:
- Translation, summarization, QA
- Code generation (GitHub Copilot)
- Creative writing, dialogue
- Arithmetic, reasoning (limited)

Limitations:
- Expensive ($4M training cost)
- Cannot update knowledge (fixed training data)
- Hallucinations (generates plausible but false info)
```

**GPT-4 (2023):**
```
Parameters: Unknown (estimated 1T+)
Multimodal: Text + images
Improved: Reasoning, factuality, safety
API-only: Not open-sourced

Capabilities:
- Pass bar exam (90th percentile)
- Explain memes, diagrams
- Write complex code
- Lengthy context (32K tokens)
```

### Prompting Strategies

**Zero-Shot:**
```
Prompt: "Translate to Spanish: I love machine learning"
Output: "Me encanta el aprendizaje automático"

No examples, just instruction
Works for simple tasks
```

**Few-Shot (In-Context Learning):**
```
Prompt:
"Translate English to Spanish:
English: Hello → Spanish: Hola
English: Goodbye → Spanish: Adiós
English: I love cats →"

Output: "Spanish: Amo a los gatos"

Provide examples in prompt
Model learns pattern from examples
No parameter updates!
```

**Chain-of-Thought Prompting:**
```
Standard prompt:
"What is 35 × 47?"
Output: 1645 (often wrong)

Chain-of-thought prompt:
"What is 35 × 47? Let's solve step-by-step:
35 × 40 = 1400
35 × 7 = 245
1400 + 245 = 1645"

Output: Correct answer with reasoning

Improves: Math, reasoning, multi-step problems
```

### GPT Limitations

**1. Hallucinations:**
```
Generates confident-sounding but false information
"Who won the 2024 World Cup?" → Makes up answer
No knowledge of what it doesn't know
```

**2. Knowledge Cutoff:**
```
Training data up to certain date
No awareness of recent events
Cannot update without retraining
```

**3. No Memory:**
```
Each conversation is isolated
Forgets previous conversations
No long-term learning from interactions
```

**4. Reasoning Limitations:**
```
Struggles with complex logic
Pattern matching, not true reasoning
Can be fooled by adversarial prompts
```

**5. Expensive:**
```
Training: Millions of dollars
Inference: High compute cost
Not accessible for small organizations
```

---

## 10. Common NLP Tasks

### 10.1 Text Classification

**Task:** Assign label to entire text

**Examples:**
- Sentiment analysis (positive/negative/neutral)
- Topic classification (sports/politics/tech)
- Spam detection
- Intent classification (customer support)

**Approach with BERT:**
```
Input: "This movie was terrible!"
BERT encoding → [CLS] token embedding
Linear layer → Softmax
Output: [negative: 0.92, neutral: 0.05, positive: 0.03]
```

**Metrics:**
- Accuracy (balanced classes)
- F1 score (imbalanced classes)
- Precision/Recall

### 10.2 Named Entity Recognition (NER)

**Task:** Identify and classify entities in text

**Example:**
```
Text: "Apple CEO Tim Cook announced new iPhone in Cupertino"
Entities:
- Apple: ORGANIZATION
- Tim Cook: PERSON
- iPhone: PRODUCT
- Cupertino: LOCATION
```

**Approach:**
Token classification - label each word
```
Apple    → B-ORG (beginning of organization)
CEO      → O (not an entity)
Tim      → B-PER (beginning of person)
Cook     → I-PER (inside person entity)
announced → O
new      → O
iPhone   → B-PROD
in       → O
Cupertino → B-LOC
```

**Tagging Scheme (BIO):**
- B: Beginning of entity
- I: Inside entity
- O: Outside (not an entity)

### 10.3 Question Answering

**Task:** Answer questions based on context

**Example:**
```
Context: "The Eiffel Tower is located in Paris, France. It was built in 1889."
Question: "Where is the Eiffel Tower?"
Answer: "Paris, France"
```

**Approach (BERT-based):**
```
Input: [CLS] Question [SEP] Context [SEP]
Output: Start and end positions in context
Extract: Text span between positions
```

**Types:**
- Extractive: Answer is span in context (SQuAD)
- Abstractive: Generate answer (may not be exact span)
- Open-domain: No context provided, model must retrieve

### 10.4 Text Summarization

**Task:** Create shorter version preserving key information

**Types:**

**Extractive:**
```
Select important sentences from document
Concatenate them
Fast, grammatical (uses original sentences)
```

**Abstractive:**
```
Generate new sentences (like human summary)
May paraphrase, use different words
Requires seq2seq model (T5, BART)
```

**Metrics:**
- ROUGE (overlap with reference summary)
- BLEU (machine translation metric, adapted)
- Human evaluation (ultimately most important)

### 10.5 Machine Translation

**Task:** Translate text from one language to another

**Example:**
```
English: "I love machine learning"
French: "J'aime l'apprentissage automatique"
```

**Modern Approach: Transformers**
```
Encoder: Process source language
Decoder: Generate target language (word by word)
Attention: Align source and target words
```

**Challenges:**
- Word order differences (English vs Japanese)
- Idioms don't translate literally
- Cultural context
- Low-resource languages (limited training data)

---

## 11. Interview Questions

### Q1: "Explain the difference between BERT and GPT. When would you use each?"

**Good Answer:**

"BERT and GPT are both transformer-based but designed for different use cases:

**BERT (Encoder-only, Bidirectional):**
- Architecture: Encoder stack only
- Attention: Bidirectional (sees full context)
- Pre-training: Masked language modeling (predict masked words)
- Strengths: Understanding tasks (classification, NER, QA)
- Example: Given 'I love cats', can understand context from both sides

**Use BERT when:**
- Classification tasks (sentiment, topic, intent)
- Named entity recognition
- Question answering (extractive)
- Need to understand full context
- Don't need to generate text

**GPT (Decoder-only, Unidirectional):**
- Architecture: Decoder stack only
- Attention: Causal/autoregressive (sees only left context)
- Pre-training: Next-word prediction (language modeling)
- Strengths: Generation tasks (text completion, dialogue, code)
- Example: Given 'I love', predicts 'cats' (cannot see future words)

**Use GPT when:**
- Text generation (articles, stories, code)
- Dialogue systems (chatbots)
- Few-shot learning (learn from prompt examples)
- Creative tasks
- Translation, summarization (with prompting)

**Real-World Decision:**
For customer support system:
- Intent classification: BERT (understands full query)
- Response generation: GPT (generates natural replies)
- Combined pipeline: BERT → classifier → GPT → responder

**Trade-offs:**
- BERT: Better for understanding, needs fine-tuning per task
- GPT: More versatile (prompting), but expensive and may hallucinate
- Modern trend: GPT-style models dominating (ChatGPT, GPT-4) due to few-shot capabilities"

---

### Q2: "How would you build a sentiment analysis system for customer reviews with 10K labeled examples?"

**Good Answer:**

"I'd use transfer learning with BERT for best results with limited data:

**Approach:**

**1. Data Preparation:**
```
- Split: 8K train, 1K validation, 1K test
- Text cleaning: Remove HTML, fix encoding
- Minimal preprocessing: Keep punctuation (affects sentiment!)
- Tokenization: BERT WordPiece tokenizer
- No need for lowercasing, stop word removal (BERT handles this)
```

**2. Model Selection:**
```
Start with pre-trained BERT-base (110M parameters):
- Pre-trained on English Wikipedia + BooksCorpus
- Already understands language structure
- Transfer learning reduces data needs

Alternative: DistilBERT (66M parameters)
- 40% smaller, 60% faster
- 95% of BERT performance
- Better for production deployment
```

**3. Fine-Tuning:**
```
Architecture:
- Pre-trained BERT
- Dropout (0.1)
- Linear layer (768 → 3 classes: positive/neutral/negative)
- Softmax

Training:
- Freeze first 8 layers (keep general features)
- Unfreeze last 4 layers (adapt to sentiment)
- Learning rate: 2e-5 (low, avoid destroying pre-trained weights)
- Epochs: 3-4 (more causes overfitting with small data)
- Batch size: 16-32
- Optimizer: AdamW
```

**4. Data Augmentation (Critical with 10K examples):**
```
- Back-translation: English → French → English (creates paraphrases)
- Synonym replacement: "good" → "great", "excellent"
- Random insertion/deletion (with caution, may change sentiment)
- Mixup: Mix embeddings of examples (if labels similar)

Effectively doubles dataset size
```

**5. Evaluation:**
```
Metrics:
- Accuracy (if balanced classes)
- F1 score per class (if imbalanced)
- Confusion matrix (understand errors)

Expected performance:
- Baseline (no transfer learning): 70-75%
- BERT fine-tuning: 85-90%
- With augmentation: 90-92%
```

**6. Error Analysis:**
```
Common issues:
- Sarcasm: "Oh great, another bug" (marked positive, actually negative)
- Mixed sentiment: "Food good, service terrible" (hard to classify)
- Domain-specific language: "sick" (negative in health, positive in slang)

Solutions:
- Collect more examples of edge cases
- Multi-label classification (not single positive/negative)
- Ensemble with rule-based system (catch sarcasm patterns)
```

**7. Production Deployment:**
```
If latency is issue (BERT ~100ms):
- Distillation: Compress BERT into smaller model (DistilBERT)
- Quantization: FP32 → INT8 (4× faster)
- ONNX runtime: Optimized inference
- Caching: Cache frequent reviews

Target: <50ms P95 latency
```

**Trade-offs:**
- More data collection vs better architecture (data usually wins)
- Accuracy vs latency (BERT vs simpler model)
- Fine-tuning all layers vs partial (partial prevents overfitting)

Real example: At previous company, achieved 88% F1 with 5K examples using this approach. Key was aggressive data augmentation and careful hyperparameter tuning."

---

### Q3: "Explain attention mechanism. Why is it important in transformers?"

**Good Answer:**

"Attention is the core innovation that makes transformers work:

**The Problem Attention Solves:**

In sequence-to-sequence tasks (translation, summarization), RNNs compress entire input into single fixed-size vector (bottleneck). Long sequences lose information.

Example: Translate English to French:
'The cat that scared the dog that chased the mouse ran away'
→ By word 15, RNN has forgotten 'cat' is the subject

**Attention Mechanism:**

Instead of single context vector, compute weighted average over ALL input states:

```
When decoding word i:
1. Query: What we want to decode
2. Keys: All encoder states
3. Values: All encoder states
4. Attention weights: How relevant each input word is
5. Output: Weighted sum of values
```

**Formula:**
```
Attention(Q, K, V) = softmax(Q·K^T / √d) · V

Q·K^T: Similarity between query and each key (dot product)
√d: Scaling factor (stabilizes gradients)
softmax: Convert to probability distribution (sum to 1)
·V: Weighted sum of values
```

**Concrete Example:**

Translation: 'I love cats' → 'J'aime les chats'

When generating 'chats':
- Query: Decoder state for 'chats'
- Keys: ['I', 'love', 'cats']
- Attention weights: [0.1, 0.2, 0.7] (attends strongly to 'cats')
- Output: 0.1×embed('I') + 0.2×embed('love') + 0.7×embed('cats')

**Why Attention is Critical:**

**1. No Information Bottleneck:**
- Access to all input states (not compressed into single vector)
- Can selectively focus on relevant parts
- Perfect for long sequences

**2. Long-Range Dependencies:**
- Direct connections: Every output can attend to every input
- Path length = 1 (vs path length = sequence_length in RNN)
- No vanishing gradients

**3. Parallelizable:**
- All attention computations independent
- Unlike RNN (must process sequentially)
- 10-100× faster training

**4. Interpretable:**
- Attention weights show what model focuses on
- Useful for debugging ('Is it attending to right words?')
- Build trust with stakeholders

**Self-Attention in Transformers:**

Words attend to other words in same sentence:
'The cat sat on the mat'

'sat' attends to:
- 'cat' (0.6) - who sat?
- 'mat' (0.3) - sat on what?
- 'on' (0.1) - spatial relationship

Every word gets context-aware representation

**Multi-Head Attention:**

Use multiple attention mechanisms in parallel:
- Head 1: Syntactic relationships (subject-verb)
- Head 2: Semantic relationships (cat-animal)
- Head 3: Positional (nearby words)
- ...
- Concatenate all heads → richer representation

**Impact:**

Attention enabled:
- Transformers (BERT, GPT)
- Long document processing
- Machine translation breakthrough
- State-of-the-art across NLP tasks

Now used beyond NLP: Vision Transformers, protein folding (AlphaFold), recommender systems."

---

### Q4: "You have 1 million customer support tickets. How would you build an automatic categorization system?"

**Good Answer:**

"I'd approach this as a multi-class text classification problem with semi-supervised learning:

**Problem Analysis:**

**Data:**
- 1M tickets (large dataset - good!)
- But likely unlabeled or partially labeled
- Imbalanced: Common issues (password reset) >> rare issues (billing error)

**Requirements:**
- High accuracy (affects customer experience)
- Explainability (support agents need to trust system)
- Low latency (<100ms for real-time routing)
- Handle new categories (new products → new issues)

**Solution Architecture:**

**Phase 1: Data Preparation**

**1. Label Collection:**
```
If unlabeled:
- Semi-supervised: Label 10K examples manually (1% of data)
- Active learning: Model suggests uncertain examples to label
- Weak supervision: Use rules/heuristics to auto-label ("password" → password_reset)

If partially labeled:
- Clean existing labels (inconsistencies common)
- Stratify: Ensure all categories represented
```

**2. Category Design:**
```
Work with support team to define categories:
- Not too broad: "Technical issues" (vague)
- Not too narrow: "Password reset for Gmail accounts" (too specific)
- Sweet spot: ~20-50 categories

Hierarchy useful:
Level 1: Billing, Technical, Account
Level 2: Technical → Login, Features, Bugs
```

**3. Text Preprocessing:**
```
Minimal for modern models:
- Remove PII (names, emails - privacy)
- Fix encoding issues
- Keep punctuation (affects meaning)
- BERT tokenization (subword)
```

**Phase 2: Model Development**

**Approach 1: Transfer Learning (Primary)**
```
Model: DistilBERT (faster than BERT, 95% performance)

Architecture:
- Pre-trained DistilBERT
- Dropout (0.1)
- Linear layer (768 → num_categories)
- Softmax

Fine-tuning:
- 10K labeled examples
- 3-4 epochs
- Learning rate: 2e-5
- Class weights (handle imbalance)

Expected: 85-90% accuracy
```

**Approach 2: Semi-Supervised (Use Unlabeled Data)**
```
990K unlabeled tickets are wasted resource!

Self-training:
1. Train model on 10K labeled
2. Predict on 990K unlabeled (get pseudo-labels)
3. Use high-confidence predictions (>0.9) as training data
4. Retrain with labeled + pseudo-labeled
5. Iterate

UDA (Unsupervised Data Augmentation):
- Augment unlabeled data (back-translation, paraphrasing)
- Consistency loss: Predictions should match across augmentations
- Improves robustness

Expected boost: +5-10% accuracy
```

**Phase 3: Handling Class Imbalance**

```
Problem: 70% password resets, 0.1% rare billing bugs

Solutions:
1. Class weights: Loss weighted by inverse frequency
   common class: weight = 0.1
   rare class: weight = 10.0

2. Focal loss: Focuses on hard examples
   Easy examples (high confidence): Low loss contribution
   Hard examples (uncertain): High loss contribution

3. Oversampling rare classes: Duplicate rare examples

4. Ensemble: Train separate model for rare classes
```

**Phase 4: Explainability**

```
Support agents need to trust system:

1. Attention visualization:
   - Show which words model attended to
   - "Billing, invoice, charge" → Billing_Issue

2. LIME/SHAP:
   - Local explanations (why this prediction?)
   - "Password" (0.3), "reset" (0.4), "forgot" (0.2) → Password_Reset

3. Confidence scores:
   - High confidence (>0.9): Auto-route
   - Low confidence (0.5-0.9): Human review
   - Very low (<0.5): Escalate
```

**Phase 5: Production Deployment**

**System Architecture:**
```
Ticket arrives
  ↓
Text preprocessing (< 5ms)
  ↓
Model inference (< 50ms with DistilBERT)
  ↓
Post-processing:
  - Confidence thresholding
  - Multi-label if needed (ticket covers multiple issues)
  ↓
Routing decision + explanation
  ↓
If low confidence → Human review queue
```

**Monitoring:**
```
Metrics:
- Accuracy (via human review of samples)
- Confidence distribution (detect drift)
- Category distribution (new products → new issues)
- Latency (P50, P95, P99)

Feedback loop:
- Support agents can correct predictions
- Use corrections for retraining
- Monthly model updates
```

**Phase 6: Handling New Categories**

```
New product launch → New issue types

Solutions:
1. Few-shot learning:
   - Collect 50-100 examples of new category
   - Fine-tune model (don't retrain from scratch)

2. Hierarchical classification:
   - Level 1: Broad (Technical, Billing)
   - Level 2: Specific (within Technical)
   - New categories added at Level 2

3. "Other" category:
   - Catch-all for new issues
   - Human review → identify patterns → create new category
```

**Expected Results:**

```
Accuracy: 88-92%
Latency: 40-60ms P95
Cost savings: 60% tickets auto-routed (no human in loop)
Customer satisfaction: Faster routing → faster resolution
```

**Real Example:**
Previous company had 500K tickets/year. This approach achieved 90% accuracy, reduced routing time from 2 hours (human) to <1 minute (automated). Key was semi-supervised learning (leveraged unlabeled data) and active learning (strategically selected examples to label)."

---

## Summary: Key Takeaways

**1. Text Representation Evolution:**
- One-hot → Bag-of-words → TF-IDF → Word2Vec → BERT embeddings
- Modern: Contextual embeddings (BERT) - same word, different meanings in different contexts

**2. Word Embeddings:**
- Word2Vec: Learn from context (distributional hypothesis)
- FastText: Subword embeddings (handles OOV)
- Both surpassed by contextual embeddings

**3. RNNs → Transformers:**
- RNNs: Sequential, slow, vanishing gradients
- Transformers: Parallel, fast, no vanishing gradients
- Attention is key innovation

**4. Pre-training + Fine-tuning:**
- BERT: Bidirectional, understanding tasks
- GPT: Unidirectional, generation tasks
- Transfer learning is standard (like ImageNet for CV)

**5. Modern NLP:**
- Minimal preprocessing (models learn from raw text)
- Subword tokenization (handles OOV)
- Large language models (GPT-3, GPT-4)
- Few-shot learning (prompting, no fine-tuning)

**6. Common Tasks:**
- Classification (sentiment, topic, intent)
- NER (entity extraction)
- QA (span extraction or generation)
- Summarization (extractive or abstractive)
- Translation (seq2seq with attention)

**For Interviews:**
- Understand transformer architecture (attention, positional encoding)
- Know BERT vs GPT trade-offs
- Transfer learning strategies (freeze layers, fine-tuning)
- Handling limited labeled data (augmentation, semi-supervised)
- Production considerations (latency, explainability, monitoring)
- Show systems thinking (data pipeline, model, deployment, feedback loop)

---

**END OF DAY 5**

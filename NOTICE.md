# Third-party components and distribution

This repository contains the V10 assistant, its local web facade and the evaluation pipeline. It does
not include private documents, conversation traces, benchmark corpora, prior
assistant variants, virtual environments or downloaded model weights.

The application source is released under the MIT License (see `LICENSE`).
That license covers this repository's code only; it does not relicense the
third-party models or dependencies listed below.

Python dependencies retain their upstream licenses. The multilingual E5 model is
downloaded separately from `intfloat/multilingual-e5-small`; its pinned manifest
records the revision and the model card's MIT license designation. The Ollama
model is obtained separately and retains its own model terms.

The optional MiniCheck integration has third-party notices under
`assistant_runtime/third_party/`. Those notices and the relevant model licenses
remain applicable when the optional components are installed.

The bundled onboarding document and video metadata are synthetic examples. The
video URL is a placeholder, not a hosted demonstration video.

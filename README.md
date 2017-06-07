# char-rnn
A basic, single-layer character-level RNN based on Andrej Karpathy's "vanilla" minimal RNN. This is largely a class-based transcription of the 100-line gist provided here: https://gist.github.com/karpathy/d4dee566867f8291f086.

I am very intersted in making this code much better, even adapting it to a mutliple-layer RNN. While I will eventually have to go to a framework (like Tensorflow), I am interested in working framework-free for a while.

## Intention

I want to write a RNN that studies style and writes stylistically like a given source text. Right now, the routine hits a hard loss limit of about 39.0000, even after millions of iterations. This code is currently run on a VPS sans-GPU. As I wrote above, I am absolutely open to suggestions of how I can improve this up to the limit of needing a framework.

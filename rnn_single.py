from random import uniform
import numpy as np, pickle

class RNN:
	
	def __init__(self, vocab_size, data_size):
		self.layer_size = 100
		self.seq_len = 25
		self.learn_rate = 1e-1
		self.vocab_size = vocab_size
		self.data_size = data_size
		self.last_h = None
		
		self.model = self.load()
		
		if self.model:
			self.Wxh, self.Whh, self.Why, self.Bh, self.By, self.last_h = self.model[0], self.model[1], self.model[2], self.model[3], self.model[4], self.model[5]
		else:
			self.Wxh = np.random.randn(self.layer_size, vocab_size)*0.01
			self.Whh = np.random.randn(self.layer_size, self.layer_size)*0.01
			self.Why = np.random.randn(vocab_size, self.layer_size)*0.01
			self.Bh = np.zeros((self.layer_size,1))
			self.By = np.zeros((vocab_size,1))
			
		self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
		self.mBh, self.mBy = np.zeros_like(self.Bh), np.zeros_like(self.By)

	def diff(self, inputs, target):
		xs, hs, ys, ps = {}, {}, {}, {}
		hs[-1] = np.copy(self.last_h)
		loss = 0
		
		#Forward pass
		
		for t in xrange(len(inputs)):
			xs[t] = np.zeros((self.vocab_size,1))
			xs[t][inputs[t]] = 1
			hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.Bh)
			ys[t] = np.dot(self.Why, hs[t]) + self.By
			ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
			loss += -np.log(ps[t][target[t]])
		
		#Backward pass
		
		dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
		dBh, dBy = np.zeros_like(self.Bh), np.zeros_like(self.By)
		dh_next = np.zeros_like(hs[0])
		
		for t in reversed(xrange(len(inputs))):
			dy = np.copy(ps[t])
			dy[target[t]] -= 1
			dWhy += np.dot(dy, hs[t].T)
			dBy += dy
			dh = np.dot(self.Why.T, dy) + dh_next
			dh_raw = (1 - hs[t] * hs[t]) * dh
			dBh += dh_raw
			dWxh += np.dot(dh_raw, xs[t].T)
			dWhh += np.dot(dh_raw, hs[t-1].T)
			dh_next = np.dot(self.Whh.T, dh_raw)
	
		for dparam in [dWxh, dWhh, dWhy, dBh, dBy]:
			np.clip(dparam, -5, 5, out=dparam)
		return loss, dWxh, dWhh, dWhy, dBh, dBy, hs[len(inputs)-1]
		
	def sample(self, h, seed_ix, n):
		x = np.zeros((self.vocab_size,1))
		x[seed_ix] = 1
		ixes = []
		
		for t in xrange(n):
			h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.Bh)
			y = np.dot(self.Why, h) + self.By
			p = np.exp(y)/np.sum(np.exp(y))
			ix = np.random.choice(range(self.vocab_size), p = p.ravel())
			x = np.zeros((self.vocab_size, 1))
			x[ix] = 1
			ixes.append(ix)
		
		return ixes
	
	def run(self, char_ix, ix_char, text):
		n, p = 0, 0
		smooth_loss = -np.log(1.0/self.vocab_size) * self.seq_len
		while True:
			if p + self.seq_len + 1 >= len(text) or (n == 0 and not self.model):
				self.last_h = np.zeros((self.layer_size,1))
				p = 0
			inputs = [char_ix[ch] for ch in text[p:p+self.seq_len]]
			target = [char_ix[ch] for ch in text[p+1:p+self.seq_len+1]]
			
			if n % 1000 == 0:
				sample_ix = self.sample(self.last_h, inputs[0], 200)
				txt = ''.join(ix_char[ix] for ix in sample_ix)
				print '\n %s \n' % (txt,)
		
			loss, dWxh, dWhh, dWhy, dBh, dBy, self.last_h = self.diff(inputs, target)
			
			smooth_loss = smooth_loss * 0.999 + loss * 0.001
		
			if n % 100 == 0 : print 'iter %d, loss %f' % (n, smooth_loss)
		
			for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.Bh, self.By],[dWxh, dWhh, dWhy, dBh, dBy],[self.mWxh, self.mWhh, self.mWhy, self.mBh, self.mBy]):
				mem += dparam * dparam
				param += -self.learn_rate * dparam / np.sqrt(mem + 1e-8)
			
			#if n % 10000 == 0: self.save(dWxh, dWhh, dWhy, dBh, dBy)
			if n % 10000 == 0: self.save()
			
			p += self.seq_len
			n += 1
	
	def save(self):
		pickle.dump([self.Wxh, self.Whh, self.Why, self.Bh, self.By, self.last_h],open('model.p','wb'))
		#pickle.dump([dWxh, dWhh, dWhy, dBh, dBy, self.last_h],open('model.p','wb'))
	
	def load(self):
		try: model = pickle.load(open('model.p','rb'))
		except IOError: model = None
		return model
		
def main(text):
	chars = list(set(text))
	data_size, vocab_size = len(text), len(chars)
	char_ix = {ch:i for i, ch in enumerate(chars)}
	ix_char = {i:ch for i, ch in enumerate(chars)}
	RNN(vocab_size, data_size).run(char_ix, ix_char, text)
	
	
if __name__ == "__main__":
	with open('text/gatsby.txt','r') as f:
		contents = f.read()
	f.close()
	
	main(contents)
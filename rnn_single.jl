using StatsBase

#NETWORK PROPERTIES
LAYER_SIZE = 3
SEQUENCE_LEN = 25
LEARN_RATE = 1e-1

#NETWORK WEIGHTS, BIASES
function weights(vsize)
	W_XH = randn(LAYER_SIZE,vsize)*0.01
	W_HH = randn(LAYER_SIZE,LAYER_SIZE)*0.01
	W_HY = randn(vsize,LAYER_SIZE)*0.01
	return W_XH, W_HH, W_HY
end

function biases(vsize)
	B_H = zeros(LAYER_SIZE,1)
	B_Y = zeros(vsize,1)
	return B_H, B_Y
end

#LOSS FUNCTION

function loss(inputs,targets,vsize,W_XH,W_HH,W_HY,B_H,B_Y,LAST_H)
	xs, hs, ys, ps = Dict(), Dict(), Dict(), Dict()
	hs[0] = copy(LAST_H)
	error = 0
	
	for t in 1:length(inputs)
		xs[t] = zeros(vsize,1)
		xs[t][inputs[t]] = 1
		hs[t] = tanh((W_XH*xs[t]) + (W_HH*hs[t-1]) + B_H)
		ys[t] = (W_HY*hs[t]) + B_Y
		ps[t] = exp(ys[t])/sum(exp(ys[t]))
		error += -log(ps[t][targets[t],1])
	end
	
	D_W_XH, D_W_HH, D_W_HY = zeros(W_XH), zeros(W_HH), zeros(W_HY)
	D_B_H, D_B_Y = zeros(B_H), zeros(B_Y)
	H_NEXT = zeros(hs[0])
	
	for t in reverse(1:length(inputs))
		dy = copy(ps[t])
		dy[targets[t]] -= 1
		D_W_HY += dy*hs[t].'
		D_B_Y += dy
		dh = (W_HY.'*dy) + H_NEXT
		dh_raw = (1 - hs[t].*hs[t]).*dh
		D_B_H += dh_raw
		D_W_XH += dh_raw*xs[t].'
		D_W_HH += dh_raw*hs[t-1].'
		H_NEXT = W_HH'*dh_raw
	end
	
	D_W_XH = clamp!(D_W_XH,-5,5)
	D_W_HH = clamp!(D_W_HH,-5,5)
	D_W_HY = clamp!(D_W_HY,-5,5)
	D_B_H = clamp!(D_B_H,-5,5)
	D_B_Y = clamp!(D_B_Y,-5,5)
	
	return error, D_W_XH, D_W_HH, D_W_HY, D_B_H, D_B_Y, hs[length(inputs)]
	
end

#SAMPLER

function sample(h, seed, vsize, n, W_XH, W_HH, W_HY, B_H, B_Y)
	x = zeros(vsize,1)
	x[seed] = 1
	ixes = []
	
	for t in 1:n
		h = tanh(W_XH*x + W_HH*h + B_H)
		y = W_HY*h + B_Y
		p = exp(y)/sum(exp(y))
		w = StatsBase.weights(p)
		ix = StatsBase.sample(1:vsize,w)
		x = zeros(vsize,1)
		x[ix] = 1
		append!(ixes,ix)
	end
	
	return ixes
end

#WRITE TO FILE

function tofile(n, text)
	f = open("output.txt","a")
		write(f,"----------\n$n\n----------\n\n$text\n\n")
	close(f)
end

#INITIALIZE THE NETWORK
function run(contents)
	n,p = 1,1
	chars = Set([ch for ch in contents])
	vsize = length(chars)
	ix_char = Dict(ch for ch in enumerate(chars))
	char_ix = map(reverse,ix_char)
	errs = []
	
	smooth_loss = -log(1.0/vsize) * SEQUENCE_LEN
	
	W_XH, W_HH, W_HY = weights(vsize)
	B_H, B_Y = biases(vsize)
	
	mXh, mHh, mHy, mBh, mBy = zeros(W_XH), zeros(W_HH), zeros(W_HY), zeros(B_H), zeros(B_Y)
	
	while true
		if p + SEQUENCE_LEN + 1 >= length(contents) || n == 1
			LAST_H = zeros(LAYER_SIZE,1)
			p = 1
		end

		inputs = [char_ix[ch] for ch in contents[p:p+SEQUENCE_LEN]]
		targets = [char_ix[ch] for ch in contents[p+1:p+1+SEQUENCE_LEN]]
		
		error, D_W_XH, D_W_HH, D_W_HY, D_B_H, D_B_Y, LAST_H = loss(inputs, targets, vsize, W_XH, W_HH, W_HY, B_H, B_Y, LAST_H)
		
		append!(errs,error)		
				
		if n%100 == 0
			println("$n: ", sum(errs)/length(errs))
		end
		
		if n%1000 == 0
			text = sample(LAST_H,inputs[1],vsize,200,W_XH,W_HH,W_HY,B_H,B_Y)
			tofile(n, join([ix_char[ch] for ch in text]))
		end
		
		#MANUAL WEIGHTING
		#XH
		mXh += D_W_XH.*D_W_XH
		W_XH += -LEARN_RATE * D_W_XH ./ sqrt(mXh + 1e-8)
		#HH
		mHh += D_W_HH.*D_W_HH
		W_HH += -LEARN_RATE * D_W_HH ./ sqrt(mHh + 1e-8)
		#HY
		mHy += D_W_HY.*D_W_HY
		W_HY += -LEARN_RATE * D_W_HY ./ sqrt(mHy + 1e-8)
		#BH
		mBh += D_B_H.*D_B_H
		B_H += -LEARN_RATE * D_B_H ./ sqrt(mBh + 1e-8)
		#BY
		mBy += D_B_Y.*D_B_Y
		B_Y += -LEARN_RATE * D_B_Y ./ sqrt(mBy + 1e-8)
		
		p += SEQUENCE_LEN
		n += 1
	end
	
end

#LOAD FILE

f = open("texts/gatsby_ord.txt")
	contents = readstring(f)
close(f)

y = run(contents)
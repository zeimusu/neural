package neural

import "math/rand"

//lets just assume sigmoidal neurons now and generalise later.

func deepCopyWeights(newW [][][]float64, w [][][]float64) {
	for i := range w {
		newW[i] = make([][]float64, len(w[i]))
		for j := range w[i] {
			newW[i][j] = make([]float64, len(w[i][j]))
			copy(newW[i][j], w[i][j])
		}
	}
}
func deepCopyBias(newB [][]float64, b [][]float64) {
	for i := range b {
		newB[i] = make([]float64, len(b[i]))
		copy(newB[i], b[i])
	}
}

func dCdw(layer, neuron, in int,
	inputs, desired [][]float64,
	weights [][][]float64, biases [][]float64,
) float64 {
	epsilon := 0.1
	net := MakeSigmoidNetwork(len(inputs), weights, biases)
	out := 0.0
	for i := range inputs {
		weights[layer][neuron][in] += epsilon
		C1 := SingleCost(net, inputs[i], desired[i])
		weights[layer][neuron][in] -= 2 * epsilon
		C2 := SingleCost(net, inputs[i], desired[i])
		weights[layer][neuron][in] += epsilon
		out += (C1 - C2) / (2 * epsilon)
	}
	return out / float64(len(inputs))
}

func dCdb(layer, neuron int,
	inputs, desired [][]float64, //typically about 10 in each list
	weights [][][]float64, biases [][]float64,
) float64 {
	epsilon := 0.1
	net := MakeSigmoidNetwork(len(inputs), weights, biases)
	out := 0.0
	for i := range inputs {
		biases[layer][neuron] += epsilon
		C1 := SingleCost(net, inputs[i], desired[i])
		biases[layer][neuron] -= 2 * epsilon
		C2 := SingleCost(net, inputs[i], desired[i])
		biases[layer][neuron] += epsilon
		out += (C1 - C2) / (2 * epsilon)
	}
	return out / float64(len(inputs))
}

//run a single epoch of training
func TrainSig(inputs, desired [][]float64, weights [][][]float64, biases [][]float64) ([][][]float64, [][]float64) {
	eta := 1.
	batchSize := 10

	shuffleInputs, shuffleDesired := shuffleInputsDesired(inputs, desired)

	newWeights := make([][][]float64, len(weights))
	deepCopyWeights(newWeights, weights)
	newBiases := make([][]float64, len(weights))
	deepCopyBias(newBiases, biases)

	for i := 0; i < len(inputs)-batchSize; i += batchSize {
		for l := range weights {
			for n := range weights[l] {
				for in := range weights[l][n] {
					dC := dCdw(
						l, n, in,
						shuffleInputs[i:i+batchSize], shuffleDesired[i:i+batchSize],
						weights, biases,
					)
					newWeights[l][n][in] -= eta * dC
				}
			}
		}
		for l := range biases {
			for n := range biases[l] {
				dC := dCdb(
					l, n,
					shuffleInputs[i:i+batchSize], shuffleDesired[i:i+batchSize],
					weights, biases,
				)
				newBiases[l][n] -= eta * dC
			}
		}
	}
	return newWeights, newBiases
}

//shuffle the inputs and desired values, keeping them aligned.
func shuffleInputsDesired(
	inputs, desired [][]float64,
) (
	[][]float64, [][]float64,
) {
	shuffledInputs := make([][]float64, len(inputs))
	shuffledDesired := make([][]float64, len(desired))
	perm := rand.Perm(len(inputs))
	for i, randindex := range perm {
		shuffledInputs[i] = inputs[randindex]
		shuffledDesired[i] = desired[randindex]
	}
	return shuffledInputs, shuffledDesired
}

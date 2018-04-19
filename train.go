package neural

import (
	"math/rand"
	"reflect"
)

//lets just assume sigmoidal neurons now and generalise later.

func dCdw(layer, neuron, in int,
	inputs, desired [][]float64,
	weights [][][]float64, biases [][]float64,
) float64 {
	epsilon := 0.01
	weights[layer][neuron][in] += epsilon
	net1 := MakeSigmoidNetwork(len(inputs), weights, biases)
	weights[layer][neuron][in] -= 2 * epsilon
	net2 := MakeSigmoidNetwork(len(inputs), weights, biases)
	weights[layer][neuron][in] += epsilon
	out := 0.0
	for i := range inputs {
		C1 := SingleCost(net1, inputs[i], desired[i])
		C2 := SingleCost(net2, inputs[i], desired[i])
		out += (C1 - C2) / (2 * epsilon)
	}
	return out / float64(len(inputs))
}

func dCdb(layer, neuron int,
	inputs, desired [][]float64, //typically about 10 in each list
	weights [][][]float64, biases [][]float64,
) float64 {
	epsilon := 0.01
	biases[layer][neuron] += epsilon
	net1 := MakeSigmoidNetwork(len(inputs), weights, biases)
	biases[layer][neuron] -= 2 * epsilon
	net2 := MakeSigmoidNetwork(len(inputs), weights, biases)
	biases[layer][neuron] += epsilon
	out := 0.0
	for i := range inputs {
		C1 := SingleCost(net1, inputs[i], desired[i])
		C2 := SingleCost(net2, inputs[i], desired[i])
		out += (C1 - C2) / (2 * epsilon)
	}
	return out / float64(len(inputs))
}

//run a single epoch of training
func TrainSig(inputs, desired [][]float64, weights [][][]float64, biases [][]float64) ([][][]float64, [][]float64) {
	eta := 0.1
	batchSize := 10
	shuffleInputs, shuffleDesired := shuffleInputsDesired(inputs, desired)
	newWeights := make([][][]float64, len(weights))
	reflect.Copy(reflect.ValueOf(newWeights), reflect.ValueOf(weights))
	newBiases := make([][]float64, len(weights))
	reflect.Copy(reflect.ValueOf(newBiases), reflect.ValueOf(biases))
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

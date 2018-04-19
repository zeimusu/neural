package neural

import "math/rand"

func init() {
	rand.Seed(100)
}

//return a randomlist of given length
func randList(length int, a, b float64) []float64 {
	out := make([]float64, length)
	for i := range out {
		out[i] = rand.Float64()*(b-a) + a
	}
	return out
}

//create random weights for one layer
func randLayerWeights(numNeurons, numInputs int) [][]float64 {
	out := make([][]float64, numNeurons)
	for i := range out {
		out[i] = randList(numInputs, -1, 1)
	}
	return out
}

//create the random weights for a whole network
func randNetworkWeights(layerSizes []int) [][][]float64 {
	out := make([][][]float64, len(layerSizes)-1)
	for i := 1; i < len(layerSizes); i++ {
		out[i-1] = randLayerWeights(layerSizes[i], layerSizes[i-1])
	}
	return out
}

//MakeRandomSumNetwork Creates a random network
func MakeRandomSumNetwork(layerSizes []int) (Network, [][][]float64) {
	weights := randNetworkWeights(layerSizes)
	return MakeWeightedNetwork(layerSizes[0], weights), weights
}

//Make RandomSigmoid makes a random network in whihc all the biass are zero and the weights are random
func MakeRandomSigmoid(layerSizes []int) (Network, [][][]float64, [][]float64) {
	weights := randNetworkWeights(layerSizes)
	biases := make([][]float64, len(layerSizes)-1)
	for i := 1; i < len(layerSizes); i++ {
		biases[i-1] = make([]float64, layerSizes[i])
	}
	return MakeSigmoidNetwork(layerSizes[0], weights, biases), weights, biases
}

//MakeRandomWeightedSumNeuron makes a neuron that just dots inputs and
//weights the weights are chosen at random.
func makeRandomWeightedSumNeuron(length int) *Neuron {
	weights := make([]float64, length)
	for i := range weights {
		weights[i] = rand.Float64()
	}
	return MakeWeightedSumNeuron(weights)
}

//MakeRandomWeightedAvNeuron, Same as the sum neuron, but scaling the
//result.
func makeRandomWeightedAvNeuron(length int) *Neuron {
	weights := make([]float64, length)
	for i := range weights {
		weights[i] = rand.Float64()
	}
	return MakeWeightedAvNeuron(weights)
}

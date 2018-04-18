package neural

import "math/rand"

func init() {
	rand.Seed(100)
}

func randList(length int, a, b float64) []float64 {
	out := make([]float64, length)
	for i := range out {
		out[i] = rand.Float64()*(b-a) - a
	}
	return out
}

func randLayerWeights(numNeurons, numInputs int) [][]float64 {
	out := make([][]float64, numNeurons)
	for i := range out {
		out[i] = randList(numInputs, -1, 1)
	}
	return out
}

func randNetworkWeights(layerSizes []int) [][][]float64 {
	out := make([][][]float64, len(layerSizes)-1)
	for i := 1; i < len(layerSizes); i++ {
		out[i-1] = randLayerWeights(layerSizes[i], layerSizes[i-1])
	}
	return out
}

func MakeRandomSumNetwork(layerSizes []int) (Network, [][][]float64) {
	weights := randNetworkWeights(layerSizes)
	return MakeWeightedNetwork(layerSizes[0], weights), weights
}

func MakeRandomSigmoid(layerSizes []int) (Network, [][][]float64, [][]float64) {
	weights := randNetworkWeights(layerSizes)
	biases := make([][]float64, len(layerSizes)-1)
	for i := 1; i < len(layerSizes); i++ {
		biases[i-1] = make([]float64, layerSizes[i])
	}
	return MakeSigmoidNetwork(layerSizes[0], weights, biases), weights, biases
}

func MakeRandomWeightedSumNeuron(length int) *Neuron {
	weights := make([]float64, length)
	for i := range weights {
		weights[i] = rand.Float64()
	}
	return MakeWeightedSumNeuron(weights)
}

func MakeRandomWeightedAvNeuron(length int) *Neuron {
	weights := make([]float64, length)
	for i := range weights {
		weights[i] = rand.Float64()
	}
	return MakeWeightedAvNeuron(weights)
}

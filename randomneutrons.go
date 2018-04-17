package main

import "math/rand"

func init() {
	rand.Seed(100)
}

func MakeRandomWeightedSumNeuron(length int) *neuron {
	weights := make([]float64, length)
	for i := range weights {
		weights[i] = rand.Float64()
	}
	return MakeWeightedSumNeuron(weights)
}

func MakeRandomWeightedAvNeuron(length int) *neuron {
	weights := make([]float64, length)
	for i := range weights {
		weights[i] = rand.Float64()
	}
	return MakeWeightedAvNeuron(weights)
}

func makeRandomLayer(length, inputLength int) layer {
	l := make(layer, length)
	for i := range l {
		l[i] = MakeRandomWeightedAvNeuron(inputLength)
	}
	return l
}
func MakeRandomNetwork(layerSizes []int) network {
	n := make(network, len(layerSizes))
	n[0] = make(layer, layerSizes[0])
	n.InitInput()
	for i := 1; i < len(n); i++ {
		n[i] = makeRandomLayer(layerSizes[i], layerSizes[i-1])
	}
	return n
}

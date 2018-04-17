package main

import "testing"

var layerWeights = [][]float64{
	{0.2, 0.3},
	{0, 0.5},
	{2.1, 0},
}

var networkWeights = [][][]float64{
	{
		{0.2, 0.3},
		{0, 0.5},
		{2.1, 0},
	},
	{
		{1.0, 2.0, 3.0},
		{0, 1, 0},
	},
	{
		{1, 1},
	},
}

func TestWeightLayer(t *testing.T) {
	testLayer, err := makeWeightLayer(layerWeights, 2)
	if err != nil {
		t.Errorf("makeWeightLayer returned unexpected error")
	}
	_ = testLayer
}

func TestWeightNetword(t *testing.T) {
	testNet := MakeWeightedNetwork(2, networkWeights)
	_ = testNet
}

/*
func makeWeightLayer(weights [][]float64, prevLayerSize int) (layer, error)

Each neuron has weights for each input: two inputs, 4, 4 ,1
weights =[
 [..]   [....]   [[....]]
[[..]],[[....]],
 [..]   [....]
 [..]   [....]
 ]

 len(weights)= number of hidden+output layers
 len(weights[0] = number of neurons in layer 1)
 weights[0][i] = weights for ith neuron in layer 1


func MakeWeightedNetwork(numInputs int, weights [][][]float64) network
*/

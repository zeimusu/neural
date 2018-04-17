package main

import (
	"fmt"
	"log"
)

func makeWeightLayer(weights [][]float64, prevLayerSize int) (layer, error) {
	l := make(layer, len(weights))
	for i := range l {
		if len(weights[i]) != prevLayerSize {
			return l, fmt.Errorf("Mismatch on length of weight vector, expect %v, got %v", prevLayerSize, len(weights[i]))
		}
		l[i] = MakeWeightedAvNeuron(weights[i])
	}
	return l, nil
}

/*
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

*/

func MakeWeightedNetwork(numInputs int, weights [][][]float64) network {
	var err error
	n := make(network, len(weights)+1)
	n[0] = make(layer, numInputs)
	n.InitInput()
	previousLayerSize := numInputs
	for i := 0; i < len(weights); i++ {
		n[i+1], err = makeWeightLayer(weights[i], previousLayerSize)
		if err != nil {
			log.Fatal(err)
		}
		previousLayerSize = len(weights[i])
	}
	return n
}

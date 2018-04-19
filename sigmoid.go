package neural

import (
	"fmt"
	"log"
	"math"
)

func sigma(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func makeSigmoidLayer(weights [][]float64, biases []float64, prevLayerSize int) (layer, error) {
	l := make(layer, len(weights))
	for i := range l {
		if len(weights[i]) != prevLayerSize {
			return l, fmt.Errorf("Mismatch on length of weight vector, expect %v, got %v", prevLayerSize, len(weights[i]))
		}
		l[i] = MakeSigmoidNeuron(weights[i], biases[i])
	}
	return l, nil
}

func MakeSigmoidNetwork(numInputs int, weights [][][]float64, biases [][]float64) Network {
	var err error
	if len(weights) != len(biases) {
		log.Fatalf("Lengths mismatch %v weights, %v biases", len(weights), len(biases))
	}
	net := make(Network, len(weights)+1)
	net[0] = make(layer, numInputs)
	net.initInput()
	previousLayerSize := numInputs
	for i := 0; i < len(weights); i++ {
		net[i+1], err = makeSigmoidLayer(weights[i], biases[i], previousLayerSize)
		if err != nil {
			log.Fatal(err)
		}
		previousLayerSize = len(weights[i])
	}
	return net
}

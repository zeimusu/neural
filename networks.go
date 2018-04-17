package main

import "fmt"

type layer []*neuron

type network []layer

func (net network) EvaluateNetwork() {
	for i := range net {
		if i == 0 {
			continue
		}
		net[i].EvaluateLayer(net[i-1])
	}
}

func getValues(l layer) []float64 {
	inputs := make([]float64, len(l))
	for i := range l {
		inputs[i] = l[i].value
	}
	return inputs
}

func (l layer) EvaluateLayer(inputLayer layer) {
	inputs := getValues(inputLayer)
	for _, neuron := range l {
		neuron.EvaluateNeuron(inputs)
	}
}

func (net network) GetOutputValue() float64 {
	return net[len(net)-1][0].value
}

func (net network) GetOutputValues() []float64 {
	return getValues(net[len(net)-1])
}

func (net network) SetInput(inputs []float64) {
	fmt.Println(len(inputs), len(net[0]))
	for i := 0; i < min(len(inputs), len(net[0])); i++ {
		net[0][i].value = inputs[i]
	}
}
func (net network) ClearInput() {
	for i := range net[0] {
		net[0][i].value = 0
	}
}

func (net network) InitInput() {
	for i := range net[0] {
		net[0][i] = &neuron{value: 0}
	}
}

package main

import "fmt"

type neuron struct {
	f           func([]float64) float64
	value       float64
	description string
}

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

func (n *neuron) EvaluateNeuron(inputs []float64) {
	n.value = n.f(inputs)
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

func min(a, b int) int {
	if a < b {
		return a
	} else {
		return b
	}
}

func MakeWeightedSumNeuron(weights []float64) *neuron {
	f := func(inputs []float64) float64 {
		total := 0.0
		for i := 0; i < min(len(weights), len(inputs)); i++ {
			total += inputs[i] * weights[i]
		}
		return total
	}
	return &neuron{f: f, value: 0,
		description: fmt.Sprintf("Weighted sum %v", weights)}
}

func MakeWeightedAvNeuron(weights []float64) *neuron {
	f := func(inputs []float64) float64 {
		total := 0.0
		length := min(len(weights), len(inputs))
		if length == 0 {
			return 0.0
		}
		for i := 0; i < length; i++ {
			total += inputs[i] * weights[i]
		}
		return total / float64(length)
	}
	return &neuron{f: f, value: 0,
		description: fmt.Sprintf("Weighted Av %v", weights)}
}

func main() {
	fmt.Println("vim-go")
}

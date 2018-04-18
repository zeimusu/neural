package main

import "fmt"

type neuron struct {
	f           func([]float64) float64
	value       float64
	description string
}

func (n *neuron) EvaluateNeuron(inputs []float64) {
	n.value = n.f(inputs)
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

func (n *neuron) String() string {
	return neuron.description
}

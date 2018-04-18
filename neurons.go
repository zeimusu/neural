package neural

import "fmt"

type Neuron struct {
	f           func([]float64) float64
	value       float64
	description string
}

func (n *Neuron) EvaluateNeuron(inputs []float64) {
	n.value = n.f(inputs)
}

func min(a, b int) int {
	if a < b {
		return a
	} else {
		return b
	}
}

func MakeWeightedSumNeuron(weights []float64) *Neuron {
	f := func(inputs []float64) float64 {
		total := 0.0
		for i := 0; i < min(len(weights), len(inputs)); i++ {
			total += inputs[i] * weights[i]
		}
		return total
	}
	return &Neuron{f: f, value: 0,
		description: fmt.Sprintf("Weighted sum %v", weights)}
}

func MakeWeightedAvNeuron(weights []float64) *Neuron {
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
	return &Neuron{f: f, value: 0,
		description: fmt.Sprintf("Weighted Av %v", weights)}
}

func MakeSigmoidNeuron(weights []float64, bias float64) *Neuron {
	f := func(inputs []float64) float64 {
		total := 0.0
		length := min(len(weights), len(inputs))
		if length == 0 {
			return 0.0
		}
		for i := 0; i < length; i++ {
			total += inputs[i] * weights[i]
		}
		return sigma(total + bias)
	}
	return &Neuron{f: f, value: 0,
		description: fmt.Sprintf("Sigmoid %v, bias %v", weights, bias)}
}

func MakePerceptron(weights []float64, bias float64) *Neuron {
	f := func(inputs []float64) float64 {
		total := 0.0
		length := min(len(weights), len(inputs))
		if length == 0 {
			return 0.0
		}
		for i := 0; i < length; i++ {
			total += inputs[i] * weights[i]
		}
		if total+bias > 0 {
			return 1.
		}
		return 0.
	}
	return &Neuron{f: f, value: 0.,
		description: fmt.Sprintf("Perceptron %v,bias %v", weights, bias),
	}
}

func (n *Neuron) String() string {
	return n.description
}

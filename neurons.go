package neural

import "fmt"

//A flexible neuron type.
//It has three properties, a function that maps inputs to outputs, a value
//and a human readable description.
//The function can be any mapping R^n->R.
type Neuron struct {
	f           func([]float64) float64
	value       float64
	description string
}

//EvaluateNeuron evaluates a neuron from its inputs, setting its value
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

//MakeWeightedSumNeuron
//A weighted sum neuron's output is the dot product of its inputs with
//a list of weights. This shows how a closure can be used to create a
//neuron
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

//MakeWeightedAvNeuron
//Values in the Weighted Sum Neuron tend to grow quickly.
//By dividing by the number of inputs, we keep the values small
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

//MakePerceptron
//A perceptron dots its inputs with weights, then if i.w +bias >0 it
//outputs 1, otherwise 0.
func makePerceptron(weights []float64, bias float64) *Neuron {
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
		description: fmt.Sprintf("Perceptron %v,\nbias %v", weights, bias),
	}
}

//MakeSigmoidNeuron
//This calcuates the dot product of the input and weights, but instead of
//a step function for the output, it uses a sigmoid or logistic function.
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
		description: fmt.Sprintf("Sigmoid %v,\nbias %v", weights, bias)}
}

//String interface for neuron, returns the description
func (n *Neuron) String() string {
	return n.description
}

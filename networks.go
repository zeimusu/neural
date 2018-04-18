package neural

type layer []*Neuron

type Network []layer

//EvaluateNetwork calculates the outputs of all the neurons in a network
func (net Network) EvaluateNetwork() {
	for i := range net {
		if i == 0 {
			continue
		}
		net[i].evaluateLayer(net[i-1])
	}
}

//getValues  returns the calcuated values in a layer of a network
func getValues(l layer) []float64 {
	inputs := make([]float64, len(l))
	for i := range l {
		inputs[i] = l[i].value
	}
	return inputs
}

//evaluateLayer calculates one layer in a network
func (l layer) evaluateLayer(inputLayer layer) {
	inputs := getValues(inputLayer)
	for _, Neuron := range l {
		Neuron.EvaluateNeuron(inputs)
	}
}

//GetOutputValue Returns the value of the first output neuron
//useful when there is just one output neuron
func (net Network) GetOutputValue() float64 {
	return net[len(net)-1][0].value
}

//GetOutputValues returns the values of the output neurons as a slice
func (net Network) GetOutputValues() []float64 {
	return getValues(net[len(net)-1])
}

//SetInput assigns a input from a float slice
func (net Network) SetInput(inputs []float64) {
	for i := 0; i < min(len(inputs), len(net[0])); i++ {
		net[0][i].value = inputs[i]
	}
}

//ClearInput resets the inputs to zero
func (net Network) ClearInput() {
	for i := range net[0] {
		net[0][i].value = 0
	}
}

//initInput sets up the inputs of a network
func (net Network) initInput() {
	for i := range net[0] {
		net[0][i] = &Neuron{value: 0}
	}
}

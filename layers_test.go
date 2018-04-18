package neural

import (
	"reflect"
	"testing"
)

var inputLayer = layer{
	&Neuron{value: 0.3}, &Neuron{value: 0.5}, &Neuron{value: 0.7},
}

var twoSumsLayer = layer{
	&Neuron{f: sumInputs}, &Neuron{f: sumInputs},
}

var sumAvLayer = layer{
	&Neuron{f: sumInputs}, &Neuron{f: average},
}

var outputLayer = layer{
	&Neuron{f: sumInputs},
}

func TestMakeInput(t *testing.T) {
	in := getValues(inputLayer)
	if !reflect.DeepEqual(in, []float64{0.3, 0.5, 0.7}) {
		t.Errorf("makeInput, expected %v, actual %v", []float64{0.3, 0.5, 0.7}, in)
	}
}

func TestEvaluateTwoSums(t *testing.T) {
	inputs := getValues(inputLayer)
	if !reflect.DeepEqual(inputs, []float64{0.3, 0.5, 0.7}) {
		t.Errorf("makeInput, expected %v, actual %v", []float64{0.3, 0.5, 0.7}, inputs)
	}
	twoSumsLayer.evaluateLayer(inputLayer)
	outputs := getValues(twoSumsLayer)
	expected := []float64{1.5, 1.5}
	if !reflect.DeepEqual(outputs, expected) {
		t.Errorf("twosums, expected %v, actual %v", expected, outputs)
	}
}

func TestSumAverage(t *testing.T) {
	sumAvLayer.evaluateLayer(inputLayer)
	outputs := getValues(sumAvLayer)
	expected := []float64{1.5, 0.5}
	if !reflect.DeepEqual(outputs, expected) {
		t.Errorf("sum av, expected %v, actual %v", expected, outputs)
	}
}

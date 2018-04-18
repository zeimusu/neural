package main

import (
	"reflect"
	"testing"
)

var inputLayer = layer{
	&neuron{value: 0.3}, &neuron{value: 0.5}, &neuron{value: 0.7},
}

var twoSumsLayer = layer{
	&neuron{f: sumInputs}, &neuron{f: sumInputs},
}

var sumAvLayer = layer{
	&neuron{f: sumInputs}, &neuron{f: average},
}

var outputLayer = layer{
	&neuron{f: sumInputs},
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

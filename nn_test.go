package main

import (
	"fmt"
	"reflect"
	"testing"
)

func sumInputs(inputs []float64) float64 {
	total := 0.0
	for _, x := range inputs {
		total += x
	}
	return total
}

func average(inputs []float64) float64 {
	if len(inputs) == 0 {
		return 0
	}
	total := 0.0
	for _, x := range inputs {
		total += x
	}
	return total / float64(len(inputs))
}

func TestEvaluateNeuron(t *testing.T) {
	//setup
	input1 := []float64{0.125, 0.25, 0.5}
	testNeuron1 := neuron{
		f:     sumInputs,
		value: 0,
	}
	testNeuron2 := neuron{f: average}
	//tests
	testNeuron1.EvaluateNeuron(input1)
	if testNeuron1.value != 0.875 {
		t.Errorf("testNeuron value %v, expected %v", testNeuron1.value, 0.875)
	}
	testNeuron1.EvaluateNeuron([]float64{0.0})
	if testNeuron1.value != 0.0 {
		t.Errorf("testNeuron value %v, expected %v", testNeuron1.value, 0.)
	}
	testNeuron2.EvaluateNeuron([]float64{2.0, 4.0})
	if testNeuron2.value != 3.0 {
		t.Errorf("testNeuron2 value %v, expected %v", testNeuron2.value, 2)
	}
}

func TestMakeWeightedSum(t *testing.T) {
	neuron := MakeWeightedSumNeuron([]float64{0.5, 1, 2})
	shortInput := []float64{2.0, 4.0}
	midInput := []float64{2, 4, 6}
	longInput := []float64{2, 4, 6, 8}
	shortExpect := 5.0
	midExpect := 17.0
	longExpect := 17.0

	neuron.EvaluateNeuron(shortInput)
	if neuron.value != shortExpect {
		t.Errorf("short input test, expected %v, actual %v", shortExpect, neuron.value)
	}
	neuron.EvaluateNeuron(midInput)
	if neuron.value != midExpect {
		t.Errorf("min input test, expected %v, actual %v", midExpect, neuron.value)
	}
	neuron.EvaluateNeuron(longInput)
	if neuron.value != longExpect {
		t.Errorf("long input test, expected %v, actual %v", longExpect, neuron.value)
	}
}

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
	twoSumsLayer.EvaluateLayer(inputLayer)
	outputs := getValues(twoSumsLayer)
	expected := []float64{1.5, 1.5}
	if !reflect.DeepEqual(outputs, expected) {
		t.Errorf("twosums, expected %v, actual %v", expected, outputs)
	}
}

func TestSumAverage(t *testing.T) {
	sumAvLayer.EvaluateLayer(inputLayer)
	outputs := getValues(sumAvLayer)
	expected := []float64{1.5, 0.5}
	if !reflect.DeepEqual(outputs, expected) {
		t.Errorf("sum av, expected %v, actual %v", expected, outputs)
	}
}

var net = network{
	inputLayer,
	twoSumsLayer,
	sumAvLayer,
	outputLayer,
}

func TestEvaluateNetwork(t *testing.T) {
	net.EvaluateNetwork()
	/*
		0.3    1.5    3.0    4.5
		0.5    1.5    1.5
		0.7
	*/
	if net.GetOutputValue() != 4.5 {
		for _, l := range net {
			fmt.Println(getValues(l))
		}
		t.Errorf("Expected 4.5, got %v", net.GetOutputValue())
	}
}

func TestRandomNetwork(t *testing.T) {
	n := MakeRandomNetwork([]int{2, 4, 4, 1})
	if len(n) != 4 {
		t.Errorf("Network has wrong length, expected 4, got %v", len(n))
	}
	if len(n[0]) != 2 {
		t.Errorf("input layer has length %v, expected 2", len(n[0]))
	}
	if len(n[1]) != 4 || len(n[2]) != 4 {
		t.Errorf("process layer has length %v,%v, expected 4,4", len(n[1]), len(n[2]))
	}
	if len(n[3]) != 1 {
		t.Errorf("output layer has length %v, expected 1", len(n[0]))
	}

	inputvalues := getValues(n[0])
	if !reflect.DeepEqual(inputvalues, []float64{0, 0}) {
		t.Errorf("Inputlayar not initialised to zero")
	}
	n.SetInput([]float64{0, 0})
	n.EvaluateNetwork()
	if n.GetOutputValue() != 0 {
		for _, l := range net {
			fmt.Println(getValues(l))
		}
		t.Errorf("Expected 0 output with  0 input, got %v", n.GetOutputValue())
	}
	/*
		n.SetInput([]float64{2, 2})
		n.EvaluateNetwork()
		if n.GetOutputValue() != 2 {
			for _, l := range n {
				fmt.Println(getValues(l))
			}
			t.Errorf("Expected 2 output with  2,2 input, got %v", n.GetOutputValue())
		}

		n.SetInput([]float64{4, 5})
		n.EvaluateNetwork()
		for _, l := range n {
			fmt.Println(getValues(l))
		}
		t.Log("Got %v with input 4,5", n.GetOutputValue())
	*/
}
